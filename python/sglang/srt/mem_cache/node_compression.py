"""
Node Compression Module for KV Cache Management

This module provides abstract and concrete implementations for compressing
KV cache stored in tree nodes by selecting the most important tokens based
on attention scores.

Classes:
    BaseNodeCompressor: Abstract base class defining the compression interface
    VanillaNodeCompressor: Original PyTorch-based implementation using per-head loops
    OptimizedNodeCompressor: Optimized implementation with vectorized operations
    TritonNodeCompressor: Triton kernel-based implementation (for large workloads)
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
    3. _process_budget_parameters: Budget parameter validation and processing
    """

    @classmethod
    def compress_node(
        cls,
        value: torch.Tensor,
        kv_pool: "MHATokenToKVPool",
        token_allocator: "BaseTokenToKVPoolAllocator",
        budget: int | float,
        residual_budget: int | float = 0,
        residual_clip: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache by selecting top-k tokens based on attention scores.

        Args:
            value: Cache locations tensor [N] containing token positions to compress
            kv_pool: The KV cache pool containing k_buffer, v_buffer, s_buffer, w_buffer
            token_allocator: Allocator for new KV cache locations
            budget: Number or fraction (0-1) of tokens to keep after compression
            residual_budget: Number or fraction of budget for residual tokens (default: 0)
            residual_clip: Maximum heap size for residual averaging (default: 0)

        Returns:
            Tuple of (original value tensor, new compressed cache locations tensor)

        Raises:
            ValueError: If budget parameters are invalid
        """
        size = len(value)
        if size == 0:
            return value, torch.empty(0, dtype=value.dtype, device=value.device)

        processed_budget, top_budget, processed_residual_budget, residual_heap_size = cls._process_budget_parameters(
            size, budget, residual_budget, residual_clip
        )

        if processed_budget == 0:
            return value, torch.empty(0, dtype=value.dtype, device=value.device)

        # If no compression needed, return original
        if processed_budget >= size:
            return value, value

        # Allocate new cache locations for compressed tokens
        out_cache_loc = token_allocator.alloc(processed_budget)

        # Compress each layer
        for layer_idx in range(kv_pool.layer_num):
            cls.compress_layer(
                layer_idx=layer_idx,
                kv_pool=kv_pool,
                value=value,
                out_cache_loc=out_cache_loc,
                top_budget=top_budget,
                residual_budget=processed_residual_budget,
                residual_heap_size=residual_heap_size,
            )

        return value, out_cache_loc

    @staticmethod
    @abstractmethod
    def compress_layer(
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

    @staticmethod
    def _process_budget_parameters(
        size: int,
        budget: int | float,
        residual_budget: int | float,
        residual_clip: int,
    ) -> Tuple[int, int, int, int]:
        """
        Validate and process budget parameters.

        Args:
            size: Total number of tokens to compress
            budget: Target budget (absolute count or fraction)
            residual_budget: Residual budget (absolute count or fraction of budget)
            residual_clip: Maximum residual heap size

        Returns:
            Tuple of (processed_budget, top_budget, processed_residual_budget, residual_heap_size)

        Raises:
            ValueError: If budget parameters are invalid
        """
        # Validate and convert budget
        if budget <= 0:
            raise ValueError(f"Invalid budget value: {budget=}")
        elif budget < 1:  # Fraction mode
            budget = int(size * budget)
        processed_budget = int(min(max(1, budget), size))

        # Validate and convert residual_budget
        if residual_budget < 0:
            raise ValueError(f"Invalid residual_budget value: {residual_budget=}")
        elif residual_budget < 1:  # Fraction mode
            residual_budget = int(processed_budget * residual_budget)
        processed_residual_budget = int(max(0, min(residual_budget, processed_budget)))

        if processed_residual_budget > processed_budget:
            raise ValueError(
                f"residual_budget ({processed_residual_budget}) cannot exceed budget ({processed_budget})"
            )

        top_budget = int(processed_budget - processed_residual_budget)

        # Calculate residual_heap_size
        if processed_residual_budget > 0:
            available_for_residual = size - top_budget
            if available_for_residual <= 0:
                residual_heap_size = 0
                processed_residual_budget = 0  # Cannot form any groups
            else:
                residual_heap_size = available_for_residual // processed_residual_budget
                if residual_clip > 0:
                    residual_heap_size = min(residual_heap_size, residual_clip)
        else:
            residual_heap_size = 0

        # Final alignment check
        if top_budget + processed_residual_budget != processed_budget:
            top_budget = processed_budget - processed_residual_budget

        return processed_budget, top_budget, processed_residual_budget, residual_heap_size


class VanillaNodeCompressor(BaseNodeCompressor):
    """
    Original vanilla PyTorch implementation.

    Uses per-head sequential processing with explicit Python loops.
    Simple and straightforward, but not the most efficient.

    Performance: Baseline
    Memory: Allocates temporary combined_k/combined_v buffers
    """

    @staticmethod
    def compress_layer(
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        N = len(value)
        device = value.device

        scores = kv_pool.s_buffer[layer_idx][value, :, 0]  # [N, head_num]
        head_num = scores.shape[1]
        k_values = kv_pool.k_buffer[layer_idx][value, :, :]  # [N, head_num, head_dim]
        v_values = kv_pool.v_buffer[layer_idx][value, :, :]

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
            kv_pool.k_buffer[layer_idx][out_cache_loc[:actual_budget]] = combined_k[:actual_budget]
            kv_pool.v_buffer[layer_idx][out_cache_loc[:actual_budget]] = combined_v[:actual_budget]

            # Update w_buffer
            if top_budget > 0:
                kv_pool.w_buffer[layer_idx][out_cache_loc[:top_budget]] = 0
            if residual_budget > 0 and residual_heap_size > 0:
                log_heap_size = torch.log(torch.tensor(residual_heap_size, dtype=torch.float32, device=device))
                kv_pool.w_buffer[layer_idx][out_cache_loc[top_budget:actual_budget]] = log_heap_size

class OptimizedNodeCompressor(BaseNodeCompressor):

    @staticmethod
    def compress_layer(
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        N = value.shape[0]
        device = value.device

        s_buffer = kv_pool.s_buffer[layer_idx]
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = s_buffer.shape[1]

        # Sort scores for all heads at once
        scores = s_buffer[value, :, 0]  # [N, head_num]
        _, sorted_indices = torch.sort(scores, dim=0, descending=True)  # [N, head_num]

        actual_budget = top_budget + residual_budget

        # === VECTORIZED and MEMORY-EFFICIENT: Process ALL heads simultaneously ===
        head_indices = torch.arange(head_num, device=device)

        # --- Top tokens ---
        if top_budget > 0:
            # Get the top indices relative to the `value` tensor
            top_indices_relative = sorted_indices[:top_budget, :]  # [top_budget, head_num]
            # Get the final source locations from the main buffer
            final_top_locs = value[top_indices_relative]  # [top_budget, head_num]

            # Create head indices for broadcasting
            head_range = head_indices[None, :]  # [1, head_num]
            head_range_expanded = head_range.expand(top_budget, -1)

            # Define destination locations
            out_loc_range = out_cache_loc[:top_budget, None]  # [top_budget, 1]

            # Perform a direct gather-scatter from the main buffer to the output locations
            k_buffer[out_loc_range, head_range_expanded, :] = k_buffer[final_top_locs, head_range_expanded, :]
            v_buffer[out_loc_range, head_range_expanded, :] = v_buffer[final_top_locs, head_range_expanded, :]

        # --- Residual tokens ---
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if N - top_budget >= num_residual_source:
                # Get residual indices relative to the `value` tensor: [num_residual_source, head_num]
                residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                # Sort for memory locality
                residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                # Reshape to [residual_budget, residual_heap_size, head_num]
                residual_indices_relative = residual_indices_relative.reshape(residual_budget, residual_heap_size, head_num)

                # Get final source locations from the main buffer
                final_residual_locs = value[residual_indices_relative]  # [residual_budget, heap_size, head_num]

                # Create head indices for broadcasting
                head_range_res = head_indices[None, None, :]  # [1, 1, head_num]
                head_expanded = head_range_res.expand(residual_budget, residual_heap_size, -1)

                # Gather directly from main buffer: [residual_budget, heap_size, head_num, head_dim]
                residual_k = k_buffer[final_residual_locs, head_expanded, :]
                residual_v = v_buffer[final_residual_locs, head_expanded, :]

                # Average over heap dimension: [residual_budget, head_num, head_dim]
                k_residual = residual_k.mean(dim=1)
                v_residual = residual_v.sum(dim=1)

                # Scatter to output locations
                out_loc_res = out_cache_loc[top_budget:actual_budget, None]  # [residual_budget, 1]
                head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                v_buffer[out_loc_res, head_range_scatter, :] = v_residual

        # Update w_buffer
        if actual_budget > 0:
            if top_budget > 0:
                kv_pool.w_buffer[layer_idx][out_cache_loc[:top_budget]] = 0
            if residual_budget > 0 and residual_heap_size > 0:
                log_heap_size = torch.log(torch.tensor(residual_heap_size, dtype=torch.float32, device=device))
                kv_pool.w_buffer[layer_idx][out_cache_loc[top_budget:actual_budget]] = log_heap_size


# Default compressor to use
NodeCompressor = OptimizedNodeCompressor

class UltraOptimizedNodeCompressor(BaseNodeCompressor):
    """
    Ultra-optimized implementation matching OptimizedNodeCompressor behavior.

    This implementation is identical to OptimizedNodeCompressor but serves as a
    template for future optimizations. Currently validates correctness.

    Performance: Equivalent to OptimizedNodeCompressor
    """

    @staticmethod
    def compress_layer(
        layer_idx: int,
        kv_pool: "MHATokenToKVPool",
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ) -> None:
        N = value.shape[0]
        device = value.device

        s_buffer = kv_pool.s_buffer[layer_idx]
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = s_buffer.shape[1]

        # Sort scores for all heads at once
        scores = s_buffer[value, :, 0]  # [N, head_num]
        _, sorted_indices = torch.sort(scores, dim=0, descending=True)  # [N, head_num]

        actual_budget = top_budget + residual_budget

        # Precompute head_indices (reused multiple times)
        head_indices = torch.arange(head_num, device=device)

        # --- Top tokens ---
        if top_budget > 0:
            top_indices_relative = sorted_indices[:top_budget, :]  # [top_budget, head_num]
            final_top_locs = value[top_indices_relative]  # [top_budget, head_num]

            # Direct indexing without creating intermediate expanded tensors
            head_range_expanded = head_indices[None, :].expand(top_budget, -1)
            out_loc_range = out_cache_loc[:top_budget, None]  # [top_budget, 1]

            # Direct gather-scatter
            k_buffer[out_loc_range, head_range_expanded, :] = k_buffer[final_top_locs, head_range_expanded, :]
            v_buffer[out_loc_range, head_range_expanded, :] = v_buffer[final_top_locs, head_range_expanded, :]

        # --- Residual tokens ---
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if N - top_budget >= num_residual_source:
                residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                residual_indices_relative = residual_indices_relative.reshape(residual_budget, residual_heap_size, head_num)

                final_residual_locs = value[residual_indices_relative]  # [residual_budget, heap_size, head_num]
                head_expanded = head_indices[None, None, :].expand(residual_budget, residual_heap_size, -1)

                # Gather from buffer
                residual_k = k_buffer[final_residual_locs, head_expanded, :]
                residual_v = v_buffer[final_residual_locs, head_expanded, :]

                # K uses mean, V uses sum (as per user modification)
                k_residual = residual_k.mean(dim=1)
                v_residual = residual_v.sum(dim=1)

                # Scatter to output locations
                out_loc_res = out_cache_loc[top_budget:actual_budget, None]  # [residual_budget, 1]
                head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                v_buffer[out_loc_res, head_range_scatter, :] = v_residual

        # Update w_buffer - same as OptimizedNodeCompressor (broadcasting handles shape)
        if actual_budget > 0:
            if top_budget > 0:
                kv_pool.w_buffer[layer_idx][out_cache_loc[:top_budget]] = 0
            if residual_budget > 0 and residual_heap_size > 0:
                log_heap_size = torch.log(torch.tensor(residual_heap_size, dtype=torch.float32, device=device))
                kv_pool.w_buffer[layer_idx][out_cache_loc[top_budget:actual_budget]] = log_heap_size

__all__ = [
    "BaseNodeCompressor",
    "VanillaNodeCompressor",
    "OptimizedNodeCompressor",
    "NodeCompressor",
]
