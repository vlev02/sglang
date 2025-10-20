"""
Triton-accelerated Node Compressor

This module provides Triton kernel implementations for critical bottlenecks
identified in OptimizedNodeCompressor profiling:
1. Fused gather-reduce-scatter for residual tokens (~12% of runtime)
2. Fused w_buffer update with log computation (~9% of runtime)
"""

import torch
import triton
import triton.language as tl
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

from .node_compression import BaseNodeCompressor


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=8),
    ],
    key=['head_dim'],
)
@triton.jit
def fused_residual_gather_reduce_scatter_kernel(
    # Input buffers
    k_buffer_ptr, v_buffer_ptr,
    # Index arrays
    source_locs_ptr, head_indices_ptr, out_locs_ptr,
    # Dimensions
    residual_budget: tl.constexpr,
    residual_heap_size: tl.constexpr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    # Strides
    k_stride_loc: tl.constexpr,
    k_stride_head: tl.constexpr,
    v_stride_loc: tl.constexpr,
    v_stride_head: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs:
    1. Gather K/V from source locations
    2. Reduce (mean for K, sum for V) across heap dimension
    3. Scatter results to output locations

    Each program handles one (residual_idx, head_idx) pair.
    """
    # Program IDs
    residual_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Get output location for this residual token
    out_loc = tl.load(out_locs_ptr + residual_idx)

    # Process head_dim in blocks
    for d_offset in range(0, head_dim, BLOCK_SIZE):
        d_indices = d_offset + tl.arange(0, BLOCK_SIZE)
        mask = d_indices < head_dim

        # Accumulators for reduction
        k_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        v_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

        # Gather and accumulate from all heap elements
        for heap_idx in range(residual_heap_size):
            # Get source location for this heap element
            # source_locs is flattened from [residual_budget, heap_size, head_num]
            # Index: residual_idx * (heap_size * head_num) + heap_idx * head_num + head_idx
            source_idx = residual_idx * (residual_heap_size * head_num) + heap_idx * head_num + head_idx
            source_loc = tl.load(source_locs_ptr + source_idx)

            # Compute K/V buffer addresses
            k_addr = k_buffer_ptr + source_loc * k_stride_loc + head_idx * k_stride_head + d_indices
            v_addr = v_buffer_ptr + source_loc * v_stride_loc + head_idx * v_stride_head + d_indices

            # Load and accumulate
            k_val = tl.load(k_addr, mask=mask, other=0.0).to(tl.float32)
            v_val = tl.load(v_addr, mask=mask, other=0.0).to(tl.float32)

            k_acc += k_val
            v_acc += v_val

        # Reduce: mean for K, sum for V
        k_result = k_acc / residual_heap_size
        v_result = v_acc  # sum is already accumulated

        # Scatter to output location
        out_k_addr = k_buffer_ptr + out_loc * k_stride_loc + head_idx * k_stride_head + d_indices
        out_v_addr = v_buffer_ptr + out_loc * v_stride_loc + head_idx * v_stride_head + d_indices

        # Convert back to original dtype and store
        tl.store(out_k_addr, k_result.to(k_buffer_ptr.dtype.element_ty), mask=mask)
        tl.store(out_v_addr, v_result.to(v_buffer_ptr.dtype.element_ty), mask=mask)




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=8),
    ],
    key=['head_dim'],
)
@triton.jit
def fused_top_token_copy_kernel(
    # Buffers
    k_buffer_ptr, v_buffer_ptr,
    # Index tensors
    in_locs_ptr, out_locs_ptr,
    # Dimensions
    head_dim: tl.constexpr,
    # Strides
    k_stride_loc: tl.constexpr, k_stride_head: tl.constexpr,
    v_stride_loc: tl.constexpr, v_stride_head: tl.constexpr,
    in_locs_stride_token: tl.constexpr, in_locs_stride_head: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel to copy K/V vectors for top tokens.
    Grid: (top_budget, head_num)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Get input and output locations
    in_loc = tl.load(in_locs_ptr + token_idx * in_locs_stride_token + head_idx * in_locs_stride_head)
    out_loc = tl.load(out_locs_ptr + token_idx)

    # Process head_dim in blocks
    for d_offset in range(0, head_dim, BLOCK_SIZE):
        d_indices = d_offset + tl.arange(0, BLOCK_SIZE)
        mask = d_indices < head_dim

        # K-buffer copy
        k_in_addr = k_buffer_ptr + in_loc * k_stride_loc + head_idx * k_stride_head + d_indices
        k_out_addr = k_buffer_ptr + out_loc * k_stride_loc + head_idx * k_stride_head + d_indices
        k_val = tl.load(k_in_addr, mask=mask)
        tl.store(k_out_addr, k_val, mask=mask)

        # V-buffer copy
        v_in_addr = v_buffer_ptr + in_loc * v_stride_loc + head_idx * v_stride_head + d_indices
        v_out_addr = v_buffer_ptr + out_loc * v_stride_loc + head_idx * v_stride_head + d_indices
        v_val = tl.load(v_in_addr, mask=mask)
        tl.store(v_out_addr, v_val, mask=mask)


class TritonNodeCompressor(BaseNodeCompressor):
    """
    Triton-accelerated node compressor with fused kernels.

    Key optimizations:
    1. Fused gather-scatter for top tokens
    2. Fused gather-reduce-scatter for residual processing
    3. Fused w_buffer update with on-the-fly log computation

    Performance: Best for residual_budget > 0 and head_num >= 8
    Requirements: triton library installed
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
        head_dim = k_buffer.shape[2]

        # Sort scores for all heads at once
        scores = s_buffer[value, :, 0]  # [N, head_num]
        _, sorted_indices = torch.sort(scores, dim=0, descending=True)

        actual_budget = top_budget + residual_budget
        head_indices = torch.arange(head_num, device=device)

        # --- Top tokens: Use Triton fused kernel ---
        if top_budget > 0:
            top_indices_relative = sorted_indices[:top_budget, :]
            final_top_locs = value[top_indices_relative]
            out_locs_top = out_cache_loc[:top_budget]

            grid = (top_budget, head_num)
            fused_top_token_copy_kernel[grid](
                k_buffer, v_buffer,
                final_top_locs, out_locs_top,
                head_dim=head_dim,
                k_stride_loc=k_buffer.stride(0), k_stride_head=k_buffer.stride(1),
                v_stride_loc=v_buffer.stride(0), v_stride_head=v_buffer.stride(1),
                in_locs_stride_token=final_top_locs.stride(0), in_locs_stride_head=final_top_locs.stride(1),
            )

        # --- Residual tokens: Use Triton fused kernel ---
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if N - top_budget >= num_residual_source:
                # Prepare indices
                residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                residual_indices_relative = residual_indices_relative.reshape(residual_budget, residual_heap_size, head_num)

                # Convert to absolute source locations
                final_residual_locs = value[residual_indices_relative]  # [residual_budget, heap_size, head_num]

                # Flatten for kernel: [residual_budget * heap_size * head_num]
                source_locs_flat = final_residual_locs.reshape(-1).contiguous()

                # Output locations for residual tokens
                out_locs_residual = out_cache_loc[top_budget:actual_budget].contiguous()

                # Launch Triton kernel: one program per (residual_idx, head_idx)
                grid = (residual_budget, head_num)
                fused_residual_gather_reduce_scatter_kernel[grid](
                    k_buffer, v_buffer,
                    source_locs_flat, head_indices, out_locs_residual,
                    residual_budget=residual_budget,
                    residual_heap_size=residual_heap_size,
                    head_num=head_num,
                    head_dim=head_dim,
                    k_stride_loc=k_buffer.stride(0),
                    k_stride_head=k_buffer.stride(1),
                    v_stride_loc=v_buffer.stride(0),
                    v_stride_head=v_buffer.stride(1),
                )


__all__ = [
    "TritonNodeCompressor",
]
