"""
Core Compression Engines

This module contains the fundamental single-node compression implementations.
All engines inherit from BaseCompressionEngine and implement compress_layer().

Architecture:
    BaseCompressionEngine (ABC)
    ├── VanillaCompressionEngine     - Original PyTorch implementation
    ├── OptimizedCompressionEngine   - Memory-optimized with cleanup
    ├── PreallocatingCompressionEngine - Pre-allocated buffers
    └── BatchedCompressionEngine     - Vectorized batch processing
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class BaseCompressionEngine(ABC):
    """
    Abstract base class for KV cache compression strategies.

    All compression engines implement the same interface for compressing
    a single layer's KV cache by selecting important tokens based on
    attention scores.
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
        **kwargs,
    ):
        """
        Initialize compression engine with budget configuration.

        Args:
            budget: Compression budget (num tokens or ratio < 1.0)
            page_size: Size of each cache page/node
            residual_budget: Additional residual budget for averaging
            residual_clip: Maximum heap size for residual tokens
            head_num: Number of attention heads (optional)
            head_dim: Dimension per head (optional)
            dtype: Tensor dtype (optional)
            device: Tensor device (optional)
        """
        # Validation
        if budget <= 0:
            raise ValueError(f"Invalid budget value: {budget=}")
        if page_size <= 0:
            raise ValueError(f"Invalid page_size value: {page_size=}")
        if residual_budget < 0:
            raise ValueError(f"Invalid residual_budget value: {residual_budget=}")
        if residual_clip < 0:
            raise ValueError(f"Invalid residual_clip value: {residual_clip=}")

        # Store basic config
        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        # Calculate processed budget (convert ratios to absolute values)
        size = page_size
        if budget < 1:
            budget = int(size * budget)
        self.processed_budget = int(min(max(1, budget), size))

        # Calculate residual budget
        if residual_budget < 1:
            residual_budget = int(self.processed_budget * residual_budget)
        self.processed_residual_budget = int(max(0, min(residual_budget, self.processed_budget)))

        if self.processed_residual_budget > self.processed_budget:
            raise ValueError(
                f"residual_budget ({self.processed_residual_budget}) cannot exceed budget ({self.processed_budget})"
            )

        # Calculate top budget (non-residual part)
        self.top_budget = int(self.processed_budget - self.processed_residual_budget)

        # Calculate residual heap size (how many tokens to average per residual slot)
        if self.processed_residual_budget > 0:
            available_for_residual = size - self.top_budget
            if available_for_residual <= 0:
                self.residual_heap_size = 0
                self.processed_residual_budget = 0
            else:
                self.residual_heap_size = available_for_residual // self.processed_residual_budget
                if residual_clip > 0:
                    self.residual_heap_size = min(self.residual_heap_size, residual_clip)
        else:
            self.residual_heap_size = 0

        # Ensure budget consistency
        if self.top_budget + self.processed_residual_budget != self.processed_budget:
            self.top_budget = self.processed_budget - self.processed_residual_budget

        # Log configuration
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Initialized {self.__class__.__name__}: "
            f"page_size={self.page_size}, "
            f"budget={self.processed_budget}, "
            f"top={self.top_budget}, "
            f"residual={self.processed_residual_budget}, "
            f"heap_size={self.residual_heap_size}"
        )

    def compress_node(
        self,
        value: torch.Tensor,
        kv_pool: "MHATokenToKVPool",
        out_cache_loc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a single node's KV cache across all layers.

        Args:
            value: Original KV cache indices [N]
            kv_pool: KV cache pool containing all layers
            out_cache_loc: Pre-allocated output indices [compressed_size]

        Returns:
            Tuple of (original_value, compressed_out_cache_loc)
        """
        size = len(value)
        if size == 0:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # Validate input
        if size % self.page_size:
            raise ValueError(
                f"Input value length ({size}) must be divisible by page_size ({self.page_size})"
            )

        # Early exit cases
        if self.processed_budget == 0 or self.processed_budget >= size:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        # Validate output allocation
        if out_cache_loc is None or len(out_cache_loc) < self.processed_budget:
            import warnings

            warnings.warn(
                f"Invalid out_cache_loc: expected {self.processed_budget} locations, "
                f"got {len(out_cache_loc) if out_cache_loc is not None else 'None'}. "
                f"Returning uncompressed value.",
                RuntimeWarning,
                stacklevel=2,
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

        Must be implemented by subclasses to define the compression strategy.

        Args:
            layer_idx: Index of the layer to compress
            kv_pool: KV cache pool
            value: Original KV cache indices [N]
            out_cache_loc: Output compressed indices [compressed_size]
            top_budget: Number of top-k tokens to keep
            residual_budget: Number of residual (averaged) tokens
            residual_heap_size: Heap size for residual averaging
        """
        pass


class VanillaCompressionEngine(BaseCompressionEngine):
    """
    Original PyTorch-based compression implementation.

    Uses per-head loops for token selection and compression.
    Simple and straightforward, but not optimized for performance.
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

        for node_idx in range(num_nodes):
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            N = len(node_value)

            scores = kv_pool.s_buffer[layer_idx][node_value, :, 0]
            head_num = scores.shape[1]
            k_values = kv_pool.k_buffer[layer_idx][node_value, :, :]
            v_values = kv_pool.v_buffer[layer_idx][node_value, :, :]

            combined_k = torch.zeros(
                top_budget + residual_budget,
                head_num,
                k_values.shape[2],
                dtype=k_values.dtype,
                device=device,
            )
            combined_v = torch.zeros(
                top_budget + residual_budget,
                head_num,
                v_values.shape[2],
                dtype=v_values.dtype,
                device=device,
            )

            for head_idx in range(head_num):
                head_scores = scores[:, head_idx]
                _, sorted_indices = torch.sort(head_scores, descending=True)

                top_indices = sorted_indices[:top_budget]
                combined_k[:top_budget, head_idx, :] = k_values[top_indices, head_idx, :]
                combined_v[:top_budget, head_idx, :] = v_values[top_indices, head_idx, :]

                if residual_budget > 0 and residual_heap_size > 0:
                    num_residual_source = residual_budget * residual_heap_size
                    if N - top_budget >= num_residual_source:
                        residual_source_indices = sorted_indices[
                            top_budget : top_budget + num_residual_source
                        ]
                        residual_source_indices, _ = torch.sort(residual_source_indices)
                        residual_source_indices = residual_source_indices.reshape(
                            residual_budget, residual_heap_size
                        )

                        residual_k = k_values[residual_source_indices, head_idx, :]
                        residual_v = v_values[residual_source_indices, head_idx, :]

                        combined_k[top_budget:, head_idx, :] = residual_k.mean(dim=1)
                        combined_v[top_budget:, head_idx, :] = residual_v.sum(dim=1)

            actual_budget = top_budget + residual_budget
            if actual_budget > 0:
                k_buffer[node_out_cache_loc[:actual_budget]] = combined_k[:actual_budget]
                v_buffer[node_out_cache_loc[:actual_budget]] = combined_v[:actual_budget]


class OptimizedCompressionEngine(BaseCompressionEngine):
    """
    Memory-optimized compression engine with explicit cleanup.

    Adds explicit tensor deletion and periodic cache clearing to reduce
    memory fragmentation during long compression sessions.
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

        for node_idx in range(num_nodes):
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            N = node_value.shape[0]

            scores = s_buffer[node_value, :, 0]
            _, sorted_indices = torch.sort(scores, dim=0, descending=True)

            if top_budget > 0:
                top_indices_relative = sorted_indices[:top_budget, :]
                final_top_locs = node_value[top_indices_relative]

                head_range_expanded = head_indices[None, :].expand(top_budget, -1)
                out_loc_range = node_out_cache_loc[:top_budget, None]

                k_buffer[out_loc_range, head_range_expanded, :] = k_buffer[
                    final_top_locs, head_range_expanded, :
                ]
                v_buffer[out_loc_range, head_range_expanded, :] = v_buffer[
                    final_top_locs, head_range_expanded, :
                ]

                del top_indices_relative, final_top_locs, head_range_expanded, out_loc_range

            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size
                if N - top_budget >= num_residual_source:
                    residual_indices_relative = sorted_indices[
                        top_budget : top_budget + num_residual_source, :
                    ]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.reshape(
                        residual_budget, residual_heap_size, head_num
                    )

                    final_residual_locs = node_value[residual_indices_relative]
                    head_expanded = head_indices[None, None, :].expand(
                        residual_budget, residual_heap_size, -1
                    )

                    residual_k = k_buffer[final_residual_locs, head_expanded, :]
                    residual_v = v_buffer[final_residual_locs, head_expanded, :]

                    k_residual = residual_k.mean(dim=1)
                    v_residual = residual_v.sum(dim=1)

                    out_loc_res = node_out_cache_loc[top_budget:actual_budget, None]
                    head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                    k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                    v_buffer[out_loc_res, head_range_scatter, :] = v_residual

                    del residual_indices_relative, final_residual_locs, head_expanded
                    del residual_k, residual_v, k_residual, v_residual
                    del out_loc_res, head_range_scatter

            del scores, sorted_indices

        del head_indices

        # Periodic cache cleanup
        if device.type == "cuda" and layer_idx % 4 == 3:
            torch.cuda.empty_cache()


class PreallocatingCompressionEngine(BaseCompressionEngine):
    """
    Pre-allocating compression engine with reusable GPU buffers.

    Allocates fixed-size buffers once during initialization and reuses
    them across all compression operations, reducing allocation overhead.

    Requires head_num, head_dim, dtype, and device to be specified.
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
        super().__init__(
            budget=budget,
            page_size=page_size,
            residual_budget=residual_budget,
            residual_clip=residual_clip,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        # Pre-allocate buffers for top-k tokens
        if self.top_budget > 0:
            self.top_k_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )
            self.top_v_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )

        # Pre-allocate buffers for residual tokens
        if self.processed_residual_budget > 0 and self.residual_heap_size > 0:
            self.residual_k_buffer = torch.empty(
                (self.processed_residual_budget, self.residual_heap_size, head_num, head_dim),
                dtype=dtype,
                device=device,
            )
            self.residual_v_buffer = torch.empty(
                (self.processed_residual_budget, self.residual_heap_size, head_num, head_dim),
                dtype=dtype,
                device=device,
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
        device = value.device
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = k_buffer.shape[1]

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

        for node_idx in range(num_nodes):
            node_value_start = node_idx * self.page_size
            node_value_end = node_value_start + self.page_size
            node_value = value[node_value_start:node_value_end]

            node_out_start = node_idx * self.processed_budget
            node_out_end = node_out_start + self.processed_budget
            node_out_cache_loc = out_cache_loc[node_out_start:node_out_end]

            scores = kv_pool.s_buffer[layer_idx][node_value, :, 0]
            _, sorted_indices = torch.sort(scores, dim=0, descending=True)

            head_indices = torch.arange(head_num, device=device)
            actual_budget = top_budget + residual_budget

            if top_budget > 0:
                top_indices_relative = sorted_indices[:top_budget, :]
                final_top_locs = node_value[top_indices_relative]
                head_range_expanded = head_indices[None, :].expand(top_budget, -1)

                self.top_k_buffer.copy_(k_buffer[final_top_locs, head_range_expanded, :])
                self.top_v_buffer.copy_(v_buffer[final_top_locs, head_range_expanded, :])

                out_loc_range = node_out_cache_loc[:top_budget, None]
                k_buffer[out_loc_range, head_range_expanded, :] = self.top_k_buffer
                v_buffer[out_loc_range, head_range_expanded, :] = self.top_v_buffer

            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size

                if node_value.shape[0] - top_budget >= num_residual_source:
                    residual_indices_relative = sorted_indices[
                        top_budget : top_budget + num_residual_source, :
                    ]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.view(
                        residual_budget, residual_heap_size, head_num
                    )

                    final_residual_locs = node_value[residual_indices_relative]
                    head_expanded = head_indices[None, None, :].expand(
                        residual_budget, residual_heap_size, -1
                    )

                    self.residual_k_buffer.copy_(k_buffer[final_residual_locs, head_expanded, :])
                    self.residual_v_buffer.copy_(v_buffer[final_residual_locs, head_expanded, :])

                    k_residual = self.residual_k_buffer.mean(dim=1)
                    v_residual = self.residual_v_buffer.sum(dim=1)

                    out_loc_res = node_out_cache_loc[top_budget:actual_budget, None]
                    head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                    k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                    v_buffer[out_loc_res, head_range_scatter, :] = v_residual


class BatchedCompressionEngine(PreallocatingCompressionEngine):
    """
    Vectorized batch compression engine for maximum GPU utilization.

    Processes multiple nodes simultaneously using batched tensor operations,
    eliminating Python loops and maximizing GPU throughput.

    Extends PreallocatingCompressionEngine for buffer reuse.
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
        """Compress layer for multiple nodes simultaneously using batched operations."""
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        s_buffer = kv_pool.s_buffer[layer_idx]

        num_nodes = value.shape[0] // self.page_size
        if num_nodes == 0:
            return

        head_num = self.head_num
        actual_budget = top_budget + residual_budget

        # Reshape inputs for batched processing: [B, ...]
        node_values = value.view(num_nodes, self.page_size)
        node_out_cache_locs = out_cache_loc.view(num_nodes, self.processed_budget)

        # Batched score gathering and sorting: [B, N, H]
        scores = s_buffer[node_values, :, 0]
        _, sorted_indices = torch.sort(scores, dim=1, descending=True)

        # Batched top-k processing
        if top_budget > 0:
            # Get relative indices for top-k tokens: [B, T, H]
            top_indices_relative = sorted_indices[:, :top_budget, :]

            # Gather absolute token locations: [B, T, H]
            final_top_locs = torch.gather(
                node_values.unsqueeze(2).expand(-1, -1, head_num), 1, top_indices_relative
            )

            # Flatten for gather: [B*H, T]
            final_top_locs_flat = final_top_locs.permute(0, 2, 1).reshape(
                num_nodes * head_num, top_budget
            )

            # Create head index tensor: [B*H]
            head_indices_flat = torch.arange(head_num, device=self.device).repeat(num_nodes)

            # Gather K/V values: [T, B*H, D]
            top_k_gathered = k_buffer[final_top_locs_flat.T, head_indices_flat, :]
            top_v_gathered = v_buffer[final_top_locs_flat.T, head_indices_flat, :]

            # Prepare output locations: [B*H, T]
            out_locs_top = (
                node_out_cache_locs[:, :top_budget]
                .unsqueeze(1)
                .expand(-1, head_num, -1)
                .reshape(num_nodes * head_num, top_budget)
            )

            # Scatter to output
            k_buffer[out_locs_top.T, head_indices_flat, :] = top_k_gathered
            v_buffer[out_locs_top.T, head_indices_flat, :] = top_v_gathered

        # Batched residual processing
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if self.page_size - top_budget >= num_residual_source:
                # Get residual indices: [B, R*heap, H]
                residual_indices_relative = sorted_indices[
                    :, top_budget : top_budget + num_residual_source, :
                ]
                residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=1)
                # Reshape for grouping: [B, R, heap, H]
                residual_indices_relative = residual_indices_relative.view(
                    num_nodes, residual_budget, residual_heap_size, head_num
                )

                # Gather absolute locations: [B, R, heap, H]
                final_residual_locs = torch.gather(
                    node_values.view(num_nodes, 1, self.page_size, 1).expand(
                        -1, residual_budget, -1, head_num
                    ),
                    2,
                    residual_indices_relative,
                )

                # Flatten for gather: [B*H*R, heap]
                final_residual_locs_flat = final_residual_locs.permute(0, 3, 1, 2).reshape(
                    num_nodes * head_num * residual_budget, residual_heap_size
                )

                # Create head indices: [B*H*R]
                head_indices_res_flat = (
                    torch.arange(head_num, device=self.device)
                    .repeat_interleave(residual_budget)
                    .repeat(num_nodes)
                )

                # Gather residual K/V: [heap, B*H*R, D]
                residual_k_gathered = k_buffer[final_residual_locs_flat.T, head_indices_res_flat, :]
                residual_v_gathered = v_buffer[final_residual_locs_flat.T, head_indices_res_flat, :]

                # Reduce: [B*H*R, D]
                k_residual = residual_k_gathered.mean(dim=0)
                v_residual = residual_v_gathered.sum(dim=0)

                # Prepare output locations: [B*H*R]
                out_locs_res_flat = (
                    node_out_cache_locs[:, top_budget:actual_budget]
                    .unsqueeze(1)
                    .expand(-1, head_num, -1)
                    .reshape(num_nodes * head_num, residual_budget)
                    .reshape(-1)
                )

                # Scatter to output
                k_buffer[out_locs_res_flat, head_indices_res_flat, :] = k_residual
                v_buffer[out_locs_res_flat, head_indices_res_flat, :] = v_residual


__all__ = [
    "BaseCompressionEngine",
    "VanillaCompressionEngine",
    "OptimizedCompressionEngine",
    "PreallocatingCompressionEngine",
    "BatchedCompressionEngine",
]
