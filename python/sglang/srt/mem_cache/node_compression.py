"""
Node Compression Module for KV Cache Management

This module provides abstract and concrete implementations for compressing
KV cache stored in tree nodes by selecting the most important tokens based
on attention scores.

Classes:
    BaseNodeCompressor: Abstract base class defining the compression interface
    VanillaNodeCompressor: Original PyTorch-based implementation using per-head loops
    OptimizedNodeCompressor: Optimized implementation with vectorized operations
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.managers.schedule_batch import BaseTokenToKVPoolAllocator


class BaseNodeCompressor(ABC):
    """
    Abstract base class for node compression strategies.

    All compression implementations must provide:
    1. compress_node: Main entry point for compressing a node
    2. compress_layer: Layer-specific compression logic
    """

    def __init__(
        self,
        budget: int | float,
        page_size: int,
        residual_budget: int | float = 0,
        residual_clip: int = 0,
        head_num: int = None,
        head_dim: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs  # Accept any additional arguments for subclass flexibility
    ):
        """
        Initialize the compressor with budget configuration.

        Args:
            budget: Number or fraction (0-1) of tokens to keep after compression
            page_size: Expected size of input value tensor (from radix tree's page_size)
            residual_budget: Number or fraction of budget for residual tokens (default: 0)
            residual_clip: Maximum heap size for residual averaging (default: 0)
            head_num: Number of attention heads (optional, for PreallocatingNodeCompressor)
            head_dim: Dimension of each attention head (optional, for PreallocatingNodeCompressor)
            dtype: Data type of KV cache (optional, for PreallocatingNodeCompressor)
            device: Device where cache resides (optional, for PreallocatingNodeCompressor)
            **kwargs: Additional arguments for subclass flexibility

        Raises:
            ValueError: If budget parameters are invalid
        """
        if budget <= 0:
            raise ValueError(f"Invalid budget value: {budget=}")
        if page_size <= 0:
            raise ValueError(f"Invalid page_size value: {page_size=}")
        if residual_budget < 0:
            raise ValueError(f"Invalid residual_budget value: {residual_budget=}")
        if residual_clip < 0:
            raise ValueError(f"Invalid residual_clip value: {residual_clip=}")

        self.page_size = page_size

        # Store optional parameters for subclasses like PreallocatingNodeCompressor
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Process and store final budget parameters
        # This is done once at initialization instead of on every compress_node call
        size = page_size  # Input size is always page_size

        # Convert budget to absolute value
        if budget < 1:  # Fraction mode
            budget = int(size * budget)
        self.processed_budget = int(min(max(1, budget), size))

        # Convert residual_budget to absolute value
        if residual_budget < 1:  # Fraction mode
            residual_budget = int(self.processed_budget * residual_budget)
        self.processed_residual_budget = int(max(0, min(residual_budget, self.processed_budget)))

        if self.processed_residual_budget > self.processed_budget:
            raise ValueError(
                f"residual_budget ({self.processed_residual_budget}) cannot exceed budget ({self.processed_budget})"
            )

        self.top_budget = int(self.processed_budget - self.processed_residual_budget)

        # Calculate residual_heap_size
        if self.processed_residual_budget > 0:
            available_for_residual = size - self.top_budget
            if available_for_residual <= 0:
                self.residual_heap_size = 0
                self.processed_residual_budget = 0  # Cannot form any groups
            else:
                self.residual_heap_size = available_for_residual // self.processed_residual_budget
                if residual_clip > 0:
                    self.residual_heap_size = min(self.residual_heap_size, residual_clip)
        else:
            self.residual_heap_size = 0

        # Final alignment check
        if self.top_budget + self.processed_residual_budget != self.processed_budget:
            self.top_budget = self.processed_budget - self.processed_residual_budget

        # Log compression configuration
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"Initialized {self.__class__.__name__} with compression config: "
            f"page_size={self.page_size}, "
            f"processed_budget={self.processed_budget}, "
            f"top_budget={self.top_budget}, "
            f"residual_budget={self.processed_residual_budget}, "
            f"residual_heap_size={self.residual_heap_size}"
        )

    def compress_node(
        self,
        value: torch.Tensor,
        kv_pool: "MHATokenToKVPool",
        out_cache_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache by selecting top-k tokens based on attention scores.

        Args:
            value: Cache locations tensor [N] containing token positions to compress
            kv_pool: The KV cache pool containing k_buffer, v_buffer, s_buffer, w_buffer
            out_cache_loc: Pre-allocated cache locations [processed_budget] for compressed tokens
                          (allocated by caller, e.g., RadixCache)

        Returns:
            Tuple of (original value tensor, new compressed cache locations tensor)

        Raises:
            ValueError: If input value length doesn't match expected page_size
        """
        size = len(value)
        if size == 0:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # Validate input length matches expected page_size
        if size % self.page_size:
            raise ValueError(
                f"Input value length ({size}) does not match expected page_size ({self.page_size}). "
                f"All nodes should have exactly page_size tokens."
            )

        # Use precomputed budget parameters (calculated once in __init__)
        if self.processed_budget == 0:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # If no compression needed, return original
        if self.processed_budget >= size:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # Validate that out_cache_loc was provided and has correct size
        if out_cache_loc is None or len(out_cache_loc) < self.processed_budget:
            import warnings
            warnings.warn(
                f"Invalid out_cache_loc: expected {self.processed_budget} locations, "
                f"got {len(out_cache_loc) if out_cache_loc is not None else 'None'}. "
                f"Returning uncompressed value.",
                RuntimeWarning,
                stacklevel=2
            )
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # Compress each layer
        for layer_idx in range(kv_pool.layer_num):
            self.compress_layer(
                layer_idx=layer_idx,
                kv_pool=kv_pool,
                value=value,
                out_cache_loc=out_cache_loc,
                top_budget=self.top_budget,
                residual_budget=self.processed_residual_budget,
                residual_heap_size=self.residual_heap_size,
            )

        return value, out_cache_loc

    @abstractmethod
    def compress_layer(
        self,
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        """
        Compress a single layer's KV cache.

        This method must be implemented by subclasses to define the specific
        compression strategy (e.g., PyTorch operations, Triton kernels, etc.).

        Args:
            layer_idx: Index of the layer to compress
            kv_pool: The KV cache pool
            value: Original cache locations [N]
            out_cache_loc: Output cache locations [budget]
            top_budget: Number of top tokens to keep directly
            residual_budget: Number of residual compressed tokens
            residual_heap_size: Size of each residual averaging group
        """
        pass


class VanillaNodeCompressor(BaseNodeCompressor):
    """
    Original vanilla PyTorch implementation.

    Uses per-head sequential processing with explicit Python loops.
    Simple and straightforward, but not the most efficient.

    Performance: Baseline
    Memory: Allocates temporary combined_k/combined_v buffers

    Supports both single-node and multi-node batch compression:
    - Single node: value.shape[0] == page_size, out_cache_loc.shape[0] == processed_budget
    - Multi-node batch: value.shape[0] % page_size == 0, out_cache_loc.shape[0] % processed_budget == 0
    """

    def compress_layer(
        self,
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        device = value.device
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]

        # Validate that value and out_cache_loc sizes are compatible
        value_size = value.shape[0]
        out_size = out_cache_loc.shape[0]

        num_nodes, value_remainder = divmod(value_size, self.page_size)
        expected_out_nodes, out_remainder = divmod(out_size, self.processed_budget)

        if value_remainder != 0:
            raise ValueError(
                f"value size ({value_size}) must be divisible by page_size ({self.page_size})"
            )
        if out_remainder != 0:
            raise ValueError(
                f"out_cache_loc size ({out_size}) must be divisible by processed_budget ({self.processed_budget})"
            )
        if num_nodes != expected_out_nodes:
            raise ValueError(
                f"Number of nodes mismatch: value has {num_nodes} nodes, "
                f"out_cache_loc has {expected_out_nodes} nodes"
            )

        # Process each node in the batch
        for node_idx in range(num_nodes):
            # Extract indices for this node
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            N = len(node_value)

            scores = kv_pool.s_buffer[layer_idx][node_value, :, 0]  # [N, head_num]
            head_num = scores.shape[1]
            k_values = kv_pool.k_buffer[layer_idx][node_value, :, :]  # [N, head_num, head_dim]
            v_values = kv_pool.v_buffer[layer_idx][node_value, :, :]

            # Allocate temporary buffers
            combined_k = torch.zeros(
                top_budget + residual_budget, head_num, k_values.shape[2],
                dtype=k_values.dtype, device=device
            )
            combined_v = torch.zeros(
                top_budget + residual_budget, head_num, v_values.shape[2],
                dtype=v_values.dtype, device=device
            )

            # Process each head sequentially
            for head_idx in range(head_num):
                head_scores = scores[:, head_idx]
                _, sorted_indices = torch.sort(head_scores, descending=True)

                # Top tokens
                top_indices = sorted_indices[:top_budget]
                combined_k[:top_budget, head_idx, :] = k_values[top_indices, head_idx, :]
                combined_v[:top_budget, head_idx, :] = v_values[top_indices, head_idx, :]

                # Residual tokens
                if residual_budget > 0 and residual_heap_size > 0:
                    num_residual_source = residual_budget * residual_heap_size
                    if N - top_budget >= num_residual_source:
                        residual_source_indices = sorted_indices[top_budget:top_budget + num_residual_source]
                        residual_source_indices, _ = torch.sort(residual_source_indices)
                        residual_source_indices = residual_source_indices.reshape(residual_budget, residual_heap_size)

                        residual_k = k_values[residual_source_indices, head_idx, :]  # [residual_budget, heap_size, head_dim]
                        residual_v = v_values[residual_source_indices, head_idx, :]

                        combined_k[top_budget:, head_idx, :] = residual_k.mean(dim=1)
                        combined_v[top_budget:, head_idx, :] = residual_v.sum(dim=1)

            # Scatter to output locations
            actual_budget = top_budget + residual_budget
            if actual_budget > 0:
                k_buffer[node_out_cache_loc[:actual_budget]] = combined_k[:actual_budget]
                v_buffer[node_out_cache_loc[:actual_budget]] = combined_v[:actual_budget]

class OptimizedNodeCompressor(BaseNodeCompressor):
    """
    Optimized node compressor with explicit memory cleanup.

    This version adds explicit deletion of intermediate tensors to prevent
    memory accumulation during frequent compressions.

    Supports both single-node and multi-node batch compression:
    - Single node: value.shape[0] == page_size, out_cache_loc.shape[0] == processed_budget
    - Multi-node batch: value.shape[0] % page_size == 0, out_cache_loc.shape[0] % processed_budget == 0
    """

    def compress_layer(
        self,
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        device = value.device
        s_buffer = kv_pool.s_buffer[layer_idx]
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]

        # Validate that value and out_cache_loc sizes are compatible
        value_size = value.shape[0]
        out_size = out_cache_loc.shape[0]

        num_nodes, value_remainder = divmod(value_size, self.page_size)
        expected_out_nodes, out_remainder = divmod(out_size, self.processed_budget)

        if value_remainder != 0:
            raise ValueError(
                f"value size ({value_size}) must be divisible by page_size ({self.page_size})"
            )
        if out_remainder != 0:
            raise ValueError(
                f"out_cache_loc size ({out_size}) must be divisible by processed_budget ({self.processed_budget})"
            )
        if num_nodes != expected_out_nodes:
            raise ValueError(
                f"Number of nodes mismatch: value has {num_nodes} nodes, "
                f"out_cache_loc has {expected_out_nodes} nodes"
            )

        head_num = s_buffer.shape[1]
        actual_budget = top_budget + residual_budget
        head_indices = torch.arange(head_num, device=device)

        # Process each node in the batch
        for node_idx in range(num_nodes):
            # Extract indices for this node
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            N = node_value.shape[0]

            # Sort scores for all heads at once
            scores = s_buffer[node_value, :, 0]  # [N, head_num]
            _, sorted_indices = torch.sort(scores, dim=0, descending=True)  # [N, head_num]

            # --- Top tokens ---
            if top_budget > 0:
                top_indices_relative = sorted_indices[:top_budget, :]  # [top_budget, head_num]
                final_top_locs = node_value[top_indices_relative]  # [top_budget, head_num]

                # Direct indexing without creating intermediate expanded tensors
                head_range_expanded = head_indices[None, :].expand(top_budget, -1)
                out_loc_range = node_out_cache_loc[:top_budget, None]  # [top_budget, 1]

                # Direct gather-scatter
                k_buffer[out_loc_range, head_range_expanded, :] = k_buffer[final_top_locs, head_range_expanded, :]
                v_buffer[out_loc_range, head_range_expanded, :] = v_buffer[final_top_locs, head_range_expanded, :]

                # Clean up top token intermediate tensors
                del top_indices_relative, final_top_locs, head_range_expanded, out_loc_range

            # --- Residual tokens ---
            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size
                if N - top_budget >= num_residual_source:
                    residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.reshape(residual_budget, residual_heap_size, head_num)

                    final_residual_locs = node_value[residual_indices_relative]  # [residual_budget, heap_size, head_num]
                    head_expanded = head_indices[None, None, :].expand(residual_budget, residual_heap_size, -1)

                    # Gather from buffer
                    residual_k = k_buffer[final_residual_locs, head_expanded, :]
                    residual_v = v_buffer[final_residual_locs, head_expanded, :]

                    # K uses mean, V uses sum (as per user modification)
                    k_residual = residual_k.mean(dim=1)
                    v_residual = residual_v.sum(dim=1)

                    # Scatter to output locations
                    out_loc_res = node_out_cache_loc[top_budget:actual_budget, None]  # [residual_budget, 1]
                    head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                    k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                    v_buffer[out_loc_res, head_range_scatter, :] = v_residual

                    # Clean up residual intermediate tensors
                    del residual_indices_relative, final_residual_locs, head_expanded
                    del residual_k, residual_v, k_residual, v_residual
                    del out_loc_res, head_range_scatter

            # Clean up per-node tensors
            del scores, sorted_indices

        # Clean up shared tensors
        del head_indices

        # Force GPU memory cleanup periodically
        if device.type == 'cuda' and layer_idx % 4 == 3:  # Every 4th layer
            torch.cuda.empty_cache()
            
class PreallocatingNodeCompressor(BaseNodeCompressor):
    """
    An optimized, stateful node compressor that pre-allocates and reuses GPU memory
    to avoid memory fragmentation and growth during frequent compressions.

    This compressor is instantiated once with the model's configuration and reused
    for all compression operations, ensuring stable and efficient memory usage.
    """

    def __init__(
        self,
        budget: int | float,
        page_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        residual_budget: int | float = 0,
        residual_clip: int = 0,
    ):
        """
        Initialize the compressor and pre-allocate workspace buffers.

        Args:
            budget: Number or fraction (0-1) of tokens to keep.
            page_size: Expected size of input value tensor (from radix tree's page_size).
            head_num: Number of attention heads.
            head_dim: Dimension of each attention head.
            dtype: The data type of the KV cache (e.g., torch.float16).
            device: The device where the cache resides (e.g., 'cuda:0').
            residual_budget: Number or fraction of budget for residual tokens.
            residual_clip: Maximum heap size for residual averaging.
        """
        # Initialize budget parameters from the base class with all required args
        super().__init__(
            budget=budget,
            page_size=page_size,
            residual_budget=residual_budget,
            residual_clip=residual_clip,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device
        )
        
        # --- Pre-allocation of workspace buffers based on fixed budget sizes ---

        # Buffer for gathered top-k keys/values
        if self.top_budget > 0:
            self.top_k_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )
            self.top_v_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )

        # Buffer for gathered residual source keys/values. This is the largest buffer
        # and the primary cause of memory leaks in the original implementation.
        if self.processed_residual_budget > 0 and self.residual_heap_size > 0:
            self.residual_k_buffer = torch.empty(
                (self.processed_residual_budget, self.residual_heap_size, head_num, head_dim),
                dtype=dtype, device=device
            )
            self.residual_v_buffer = torch.empty(
                (self.processed_residual_budget, self.residual_heap_size, head_num, head_dim),
                dtype=dtype, device=device
            )

    def compress_layer(
        self,
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        """
        Compresses a layer using pre-allocated buffers to avoid creating large
        temporary tensors.

        Supports both single-node and multi-node batch compression:
        - Single node: value.shape[0] == page_size, out_cache_loc.shape[0] == processed_budget
        - Multi-node batch: value.shape[0] % page_size == 0, out_cache_loc.shape[0] % processed_budget == 0
        """
        device = value.device
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = k_buffer.shape[1]

        # Validate that value and out_cache_loc sizes are compatible
        value_size = value.shape[0]
        out_size = out_cache_loc.shape[0]

        num_nodes, value_remainder = divmod(value_size, self.page_size)
        expected_out_nodes, out_remainder = divmod(out_size, self.processed_budget)

        if value_remainder != 0:
            raise ValueError(
                f"value size ({value_size}) must be divisible by page_size ({self.page_size})"
            )
        if out_remainder != 0:
            raise ValueError(
                f"out_cache_loc size ({out_size}) must be divisible by processed_budget ({self.processed_budget})"
            )
        if num_nodes != expected_out_nodes:
            raise ValueError(
                f"Number of nodes mismatch: value has {num_nodes} nodes, "
                f"out_cache_loc has {expected_out_nodes} nodes"
            )

        # Process each node in the batch
        for node_idx in range(num_nodes):
            # Extract indices for this node
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            # Sort scores for all heads at once
            scores = kv_pool.s_buffer[layer_idx][node_value, :, 0]
            _, sorted_indices = torch.sort(scores, dim=0, descending=True)

            head_indices = torch.arange(head_num, device=device)
            actual_budget = top_budget + residual_budget

            # --- Top tokens (using pre-allocated buffers) ---
            if top_budget > 0:
                top_indices_relative = sorted_indices[:top_budget, :]
                final_top_locs = node_value[top_indices_relative]
                head_range_expanded = head_indices[None, :].expand(top_budget, -1)

                # NO NEW LARGE TENSORS: Gather directly into the pre-allocated buffers.
                # We use copy_() for an in-place operation.
                self.top_k_buffer.copy_(k_buffer[final_top_locs, head_range_expanded, :])
                self.top_v_buffer.copy_(v_buffer[final_top_locs, head_range_expanded, :])

                # Scatter from the buffer to the final destination
                out_loc_range = node_out_cache_loc[:top_budget, None]
                k_buffer[out_loc_range, head_range_expanded, :] = self.top_k_buffer
                v_buffer[out_loc_range, head_range_expanded, :] = self.top_v_buffer

            # --- Residual tokens (using pre-allocated buffers) ---
            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size

                # This check is important as it's possible not to have enough tokens left
                if node_value.shape[0] - top_budget >= num_residual_source:
                    residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.view(residual_budget, residual_heap_size, head_num)

                    final_residual_locs = node_value[residual_indices_relative]
                    head_expanded = head_indices[None, None, :].expand(residual_budget, residual_heap_size, -1)

                    # NO NEW LARGE TENSORS: Gather into the pre-allocated buffers
                    self.residual_k_buffer.copy_(k_buffer[final_residual_locs, head_expanded, :])
                    self.residual_v_buffer.copy_(v_buffer[final_residual_locs, head_expanded, :])

                    # Perform computation on the buffer. This creates a smaller intermediate tensor,
                    # which is acceptable as we have avoided the main large allocation.
                    k_residual = self.residual_k_buffer.mean(dim=1)
                    v_residual = self.residual_v_buffer.sum(dim=1)

                    # Scatter to output locations
                    out_loc_res = node_out_cache_loc[top_budget:actual_budget, None]
                    head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                    k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                    v_buffer[out_loc_res, head_range_scatter, :] = v_residual
                
                
class SimplifiedNodeCompressor(BaseNodeCompressor):
    """
    Simplified PyTorch implementation.
    """
    
    
    def compress_layer(
        self,
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
            
        split_ind = self.processed_budget // 2
        k_buffer[out_cache_loc[:split_ind]] = k_buffer[value[:split_ind]]
        v_buffer[out_cache_loc[:split_ind]] = v_buffer[value[:split_ind]]
        k_buffer[out_cache_loc[split_ind:]] = k_buffer[value[split_ind-self.processed_budget:]]
        v_buffer[out_cache_loc[split_ind:]] = v_buffer[value[split_ind-self.processed_budget:]]

class UltraOptimizedNodeCompressor(PreallocatingNodeCompressor):
    pass


# Default compressor to use
NodeCompressor = PreallocatingNodeCompressor

__all__ = [
    "BaseNodeCompressor",
    "VanillaNodeCompressor",
    "OptimizedNodeCompressor",
    "UltraOptimizedNodeCompressor",
    "NodeCompressor",
]