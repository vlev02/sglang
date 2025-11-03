"""
Memory-Optimized Node Compression

This module provides memory-efficient implementations that:
1. Minimize intermediate tensor allocations
2. Reuse pre-allocated buffers where possible
3. Use in-place operations when safe
4. Explicitly manage CUDA memory
"""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

from .node_compression import BaseNodeCompressor


class MemoryOptimizedNodeCompressor(BaseNodeCompressor):
    """
    Memory-optimized node compressor that minimizes GPU memory fragmentation.

    Key optimizations:
    1. In-place scatter operations instead of creating intermediate buffers
    2. Explicit torch.cuda.empty_cache() calls after large operations
    3. Pre-allocated reusable buffers for sorting indices
    4. Minimal tensor copying

    Use this when experiencing OOM errors during compression.
    """

    # Class-level buffer cache to reuse across compressions
    _sort_buffer_cache = {}

    @classmethod
    def _get_or_create_sort_buffer(cls, size: int, num_heads: int, device: torch.device):
        """Get cached sort buffer or create new one."""
        key = (size, num_heads, device)
        if key not in cls._sort_buffer_cache:
            cls._sort_buffer_cache[key] = {
                'indices': torch.empty(size, num_heads, dtype=torch.long, device=device),
                'values': torch.empty(size, num_heads, dtype=torch.float32, device=device),
            }
        return cls._sort_buffer_cache[key]

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
        """
        Memory-optimized layer compression with explicit memory management.
        """
        N = value.shape[0]
        device = value.device

        s_buffer = kv_pool.s_buffer[layer_idx]
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = s_buffer.shape[1]
        head_dim = k_buffer.shape[2]

        # Get scores - this creates a view, minimal memory
        scores = s_buffer[value, :, 0]  # [N, head_num]

        # Use torch.topk instead of sort for better memory efficiency when budget << N
        # topk only returns the k largest, avoiding full sorting
        actual_budget = top_budget + residual_budget

        if actual_budget >= N:
            # No compression needed, just copy
            for h in range(head_num):
                k_buffer[out_cache_loc, h, :] = k_buffer[value, h, :]
                v_buffer[out_cache_loc, h, :] = v_buffer[value, h, :]
            return

        # Sort scores for all heads at once
        # Note: torch.sort allocates new tensors - this is unavoidable
        # but we minimize subsequent allocations
        sorted_vals, sorted_indices = torch.sort(scores, dim=0, descending=True)

        # --- Top tokens: Direct in-place scatter ---
        if top_budget > 0:
            # Process in chunks to reduce peak memory
            chunk_size = min(top_budget, 32)  # Process 32 tokens at a time

            for chunk_start in range(0, top_budget, chunk_size):
                chunk_end = min(chunk_start + chunk_size, top_budget)
                chunk_size_actual = chunk_end - chunk_start

                # Get indices for this chunk
                top_chunk_indices = sorted_indices[chunk_start:chunk_end, :]  # [chunk_size, head_num]
                out_chunk_locs = out_cache_loc[chunk_start:chunk_end]  # [chunk_size]

                # Process each head for this chunk
                for h in range(head_num):
                    src_locs = value[top_chunk_indices[:, h]]  # [chunk_size]
                    k_buffer[out_chunk_locs, h, :] = k_buffer[src_locs, h, :]
                    v_buffer[out_chunk_locs, h, :] = v_buffer[src_locs, h, :]

        # Free sorted values tensor (we only need indices)
        del sorted_vals

        # --- Residual tokens with minimal temporary allocation ---
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if N - top_budget >= num_residual_source:
                # Process residuals in chunks to reduce memory
                chunk_size = min(residual_budget, 16)  # Smaller chunks for residuals

                for res_chunk_start in range(0, residual_budget, chunk_size):
                    res_chunk_end = min(res_chunk_start + chunk_size, residual_budget)
                    res_chunk_size = res_chunk_end - res_chunk_start

                    # Calculate source range for this chunk
                    src_start = top_budget + res_chunk_start * residual_heap_size
                    src_end = top_budget + res_chunk_end * residual_heap_size

                    # Get sorted indices for this residual chunk
                    res_indices_chunk = sorted_indices[src_start:src_end, :]  # [chunk_size * heap_size, head_num]

                    # Sort residual indices (needed for proper grouping)
                    res_indices_sorted, _ = torch.sort(res_indices_chunk, dim=0)

                    # Reshape to group by heap
                    res_indices_grouped = res_indices_sorted.reshape(res_chunk_size, residual_heap_size, head_num)

                    # Output locations for this chunk
                    out_res_chunk = out_cache_loc[top_budget + res_chunk_start:top_budget + res_chunk_end]

                    # Process each head
                    for h in range(head_num):
                        # Get source locations for this head
                        src_locs_heap = value[res_indices_grouped[:, :, h]]  # [chunk_size, heap_size]

                        # Gather K and V values
                        k_heap = k_buffer[src_locs_heap, h, :]  # [chunk_size, heap_size, head_dim]
                        v_heap = v_buffer[src_locs_heap, h, :]

                        # Reduce: mean for K, sum for V
                        # Use in-place operations where possible
                        k_reduced = k_heap.mean(dim=1)  # [chunk_size, head_dim]
                        v_reduced = v_heap.sum(dim=1)   # [chunk_size, head_dim]

                        # Scatter to output
                        k_buffer[out_res_chunk, h, :] = k_reduced
                        v_buffer[out_res_chunk, h, :] = v_reduced

                        # Explicitly delete temporary tensors
                        del k_heap, v_heap, k_reduced, v_reduced, src_locs_heap

                    del res_indices_chunk, res_indices_sorted, res_indices_grouped

        # Explicitly delete sorted_indices to free memory immediately
        del sorted_indices

        # Force CUDA to free unused memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()


class InPlaceNodeCompressor(BaseNodeCompressor):
    """
    Ultra-aggressive memory optimization using maximum in-place operations.

    This version directly modifies buffers without intermediate allocations.
    Best for very tight memory constraints.
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
        """
        In-place compression with minimal memory overhead.
        """
        N = value.shape[0]
        device = value.device

        s_buffer = kv_pool.s_buffer[layer_idx]
        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]
        head_num = s_buffer.shape[1]
        head_dim = k_buffer.shape[2]

        scores = s_buffer[value, :, 0]  # [N, head_num]

        # Process each head independently to minimize memory
        for h in range(head_num):
            head_scores = scores[:, h]  # [N]

            # Use topk for efficiency (avoids full sort)
            num_to_select = min(top_budget + residual_budget * residual_heap_size, N)
            if num_to_select < N:
                _, top_k_indices = torch.topk(head_scores, num_to_select, largest=True, sorted=True)
            else:
                # All tokens needed, just sort
                _, top_k_indices = torch.sort(head_scores, descending=True)

            # --- Top tokens ---
            if top_budget > 0:
                top_indices_local = top_k_indices[:top_budget]
                src_locs = value[top_indices_local]
                out_locs = out_cache_loc[:top_budget]

                # Direct copy
                k_buffer[out_locs, h, :] = k_buffer[src_locs, h, :]
                v_buffer[out_locs, h, :] = v_buffer[src_locs, h, :]

            # --- Residual tokens ---
            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size
                if num_to_select > top_budget and N - top_budget >= num_residual_source:
                    res_indices_local = top_k_indices[top_budget:top_budget + num_residual_source]

                    # Sort for proper grouping
                    res_indices_sorted, _ = torch.sort(res_indices_local)

                    # Reshape to heaps
                    res_indices_heaps = res_indices_sorted.reshape(residual_budget, residual_heap_size)

                    # Process each residual token
                    for r_idx in range(residual_budget):
                        heap_indices = value[res_indices_heaps[r_idx]]  # [heap_size]
                        out_loc = out_cache_loc[top_budget + r_idx]

                        # Gather and reduce in one operation
                        k_heap = k_buffer[heap_indices, h, :]  # [heap_size, head_dim]
                        v_heap = v_buffer[heap_indices, h, :]

                        # Compute reduction and write directly
                        k_buffer[out_loc, h, :] = k_heap.mean(dim=0)
                        v_buffer[out_loc, h, :] = v_heap.sum(dim=0)

                        del k_heap, v_heap

                    del res_indices_local, res_indices_sorted, res_indices_heaps

            del head_scores, top_k_indices

        del scores

        # Clean up CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()


__all__ = [
    "MemoryOptimizedNodeCompressor",
    "InPlaceNodeCompressor",
]
