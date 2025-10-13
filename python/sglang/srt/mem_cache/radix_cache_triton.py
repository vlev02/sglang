"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Triton-optimized KV cache compression for radix tree nodes.

This module provides GPU-accelerated compression kernels using Triton.
"""

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

if TYPE_CHECKING:
    from .radix_cache import TreeNode


@triton.jit
def _select_top_indices_kernel(
    # Input pointers
    scores_ptr,  # [N, head_num] - scores for all tokens
    # Output pointers
    top_indices_ptr,  # [head_num, top_budget] - selected top token indices per head
    # Dimensions
    N: tl.constexpr,
    head_num: tl.constexpr,
    top_budget: tl.constexpr,
    # Strides
    scores_stride_n,
    scores_stride_h,
    top_indices_stride_h,
    top_indices_stride_k,
):
    """
    Select top-k tokens per head based on scores.

    Each program instance handles one head.
    """
    # Get head index
    head_idx = tl.program_id(0)

    if head_idx >= head_num:
        return

    # Load scores for this head
    score_offsets = tl.arange(0, N)
    scores = tl.load(
        scores_ptr + head_idx * scores_stride_h + score_offsets * scores_stride_n,
        mask=score_offsets < N,
        other=-float('inf')
    )

    # Selection sort to find top-k indices
    # We use a simple approach: iteratively find max and mark it
    selected = tl.zeros([N], dtype=tl.int1)

    for k in range(top_budget):
        # Find max among non-selected
        max_val = -float('inf')
        max_idx = 0

        for i in range(N):
            if not selected[i] and scores[i] > max_val:
                max_val = scores[i]
                max_idx = i

        # Mark as selected
        selected[max_idx] = 1

        # Store index
        out_offset = head_idx * top_indices_stride_h + k * top_indices_stride_k
        tl.store(top_indices_ptr + out_offset, max_idx)


@triton.jit
def _select_residual_indices_kernel(
    # Input pointers
    scores_ptr,  # [N, head_num] - scores for all tokens
    top_indices_ptr,  # [head_num, top_budget] - already selected top indices
    # Output pointers
    residual_indices_ptr,  # [head_num, residual_budget * residual_heap_size] - selected residual indices
    # Dimensions
    N: tl.constexpr,
    head_num: tl.constexpr,
    top_budget: tl.constexpr,
    residual_source_count: tl.constexpr,  # residual_budget * residual_heap_size
    # Strides
    scores_stride_n,
    scores_stride_h,
    top_indices_stride_h,
    top_indices_stride_k,
    residual_indices_stride_h,
    residual_indices_stride_k,
):
    """
    Select top scoring residual tokens (excluding already selected top tokens).
    Each program instance handles one head.
    """
    head_idx = tl.program_id(0)

    if head_idx >= head_num:
        return

    # Load scores for this head
    score_offsets = tl.arange(0, N)
    scores = tl.load(
        scores_ptr + head_idx * scores_stride_h + score_offsets * scores_stride_n,
        mask=score_offsets < N,
        other=-float('inf')
    )

    # Mark top tokens as unavailable
    top_mask = tl.zeros([N], dtype=tl.int1)
    for k in range(top_budget):
        top_idx_offset = head_idx * top_indices_stride_h + k * top_indices_stride_k
        top_idx = tl.load(top_indices_ptr + top_idx_offset)
        top_mask[top_idx] = 1

    # Select top residual_source_count tokens from remaining
    selected = tl.zeros([N], dtype=tl.int1)
    temp_indices = tl.zeros([residual_source_count], dtype=tl.int32)

    for k in range(residual_source_count):
        max_val = -float('inf')
        max_idx = 0

        for i in range(N):
            if not selected[i] and not top_mask[i] and scores[i] > max_val:
                max_val = scores[i]
                max_idx = i

        selected[max_idx] = 1
        temp_indices[k] = max_idx

    # Sort temp_indices to maintain original order (bubble sort for small arrays)
    for i in range(residual_source_count):
        for j in range(residual_source_count - 1 - i):
            if temp_indices[j] > temp_indices[j + 1]:
                # Swap
                tmp = temp_indices[j]
                temp_indices[j] = temp_indices[j + 1]
                temp_indices[j + 1] = tmp

    # Store sorted indices
    for k in range(residual_source_count):
        out_offset = head_idx * residual_indices_stride_h + k * residual_indices_stride_k
        tl.store(residual_indices_ptr + out_offset, temp_indices[k])


@triton.jit
def _compress_kv_kernel(
    # Input pointers
    k_in_ptr,  # [N, head_num, head_dim] - input K cache
    v_in_ptr,  # [N, head_num, head_dim] - input V cache
    top_indices_ptr,  # [head_num, top_budget] - top token indices
    residual_indices_ptr,  # [head_num, residual_budget * residual_heap_size] - residual indices
    value_ptr,  # [N] - original cache locations
    # Output pointers
    k_out_ptr,  # [out_budget, head_num, head_dim] - output K cache
    v_out_ptr,  # [out_budget, head_num, head_dim] - output V cache
    out_cache_loc_ptr,  # [out_budget] - output cache locations
    # Dimensions
    N: tl.constexpr,
    head_num: tl.constexpr,
    head_dim: tl.constexpr,
    top_budget: tl.constexpr,
    residual_budget: tl.constexpr,
    residual_heap_size: tl.constexpr,
    out_budget: tl.constexpr,  # top_budget + residual_budget
    # Strides
    k_in_stride_n,
    k_in_stride_h,
    k_in_stride_d,
    v_in_stride_n,
    v_in_stride_h,
    v_in_stride_d,
    k_out_stride_n,
    k_out_stride_h,
    k_out_stride_d,
    v_out_stride_n,
    v_out_stride_h,
    v_out_stride_d,
    top_indices_stride_h,
    top_indices_stride_k,
    residual_indices_stride_h,
    residual_indices_stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compress KV cache by copying top tokens and averaging residual groups.
    Each program handles one output token for one head.

    Grid: (out_budget, head_num)
    """
    out_idx = tl.program_id(0)  # Output token index
    head_idx = tl.program_id(1)  # Head index

    if out_idx >= out_budget or head_idx >= head_num:
        return

    # Determine output cache location
    out_loc = tl.load(out_cache_loc_ptr + out_idx)

    # Process in blocks of head_dim
    for d_offset in range(0, head_dim, BLOCK_SIZE):
        d_indices = d_offset + tl.arange(0, BLOCK_SIZE)
        d_mask = d_indices < head_dim

        if out_idx < top_budget:
            # Top token: direct copy
            # Get source index
            top_idx_offset = head_idx * top_indices_stride_h + out_idx * top_indices_stride_k
            src_idx = tl.load(top_indices_ptr + top_idx_offset)
            src_loc = tl.load(value_ptr + src_idx)

            # Load K and V
            k_in_offset = src_loc * k_in_stride_n + head_idx * k_in_stride_h + d_indices * k_in_stride_d
            v_in_offset = src_loc * v_in_stride_n + head_idx * v_in_stride_h + d_indices * v_in_stride_d

            k_val = tl.load(k_in_ptr + k_in_offset, mask=d_mask, other=0.0)
            v_val = tl.load(v_in_ptr + v_in_offset, mask=d_mask, other=0.0)

        else:
            # Residual token: average from group
            residual_idx = out_idx - top_budget
            group_start = residual_idx * residual_heap_size

            # Accumulate values
            k_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            v_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

            for i in range(residual_heap_size):
                residual_offset = head_idx * residual_indices_stride_h + (group_start + i) * residual_indices_stride_k
                src_idx = tl.load(residual_indices_ptr + residual_offset)
                src_loc = tl.load(value_ptr + src_idx)

                k_in_offset = src_loc * k_in_stride_n + head_idx * k_in_stride_h + d_indices * k_in_stride_d
                v_in_offset = src_loc * v_in_stride_n + head_idx * v_in_stride_h + d_indices * v_in_stride_d

                k_val_i = tl.load(k_in_ptr + k_in_offset, mask=d_mask, other=0.0)
                v_val_i = tl.load(v_in_ptr + v_in_offset, mask=d_mask, other=0.0)

                k_sum += k_val_i
                v_sum += v_val_i

            # Average
            k_val = k_sum / residual_heap_size
            v_val = v_sum / residual_heap_size

        # Store to output
        k_out_offset = out_loc * k_out_stride_n + head_idx * k_out_stride_h + d_indices * k_out_stride_d
        v_out_offset = out_loc * v_out_stride_n + head_idx * v_out_stride_h + d_indices * v_out_stride_d

        tl.store(k_out_ptr + k_out_offset, k_val, mask=d_mask)
        tl.store(v_out_ptr + v_out_offset, v_val, mask=d_mask)


class NodeCompressorTriton:
    """Triton-optimized KV cache compressor for tree nodes.

    This class provides GPU-accelerated compression using Triton kernels,
    offering significant performance improvements over the PyTorch implementation.
    """

    @staticmethod
    def _process_budget_parameters(
        size: int,
        budget: int | float,
        residual_budget: int | float,
        residual_clip: int
    ) -> tuple[int, int, int, int]:
        """Validate and process budget parameters.

        Args:
            size: Size of the cache to compress
            budget: Number or fraction of tokens to keep
            residual_budget: Number or fraction for residual budget
            residual_clip: Maximum size for residual heap

        Returns:
            Tuple of (processed_budget, top_budget, processed_residual_budget, residual_heap_size)
        """
        # Validate and process budget
        if budget <= 0:
            raise ValueError(f"Invalid budget value: {budget=}")
        elif budget < 1:
            budget = int(size * budget)
        processed_budget = min(max(1, budget), size)

        # Validate and process residual budget
        if residual_budget < 0:
            raise ValueError(f"Invalid residual_budget value: {residual_budget=}")
        elif residual_budget < 1:
            residual_budget = int(processed_budget * residual_budget)
        processed_residual_budget = max(0, min(residual_budget, processed_budget))

        if processed_residual_budget > processed_budget:
            raise ValueError(
                f"residual_budget ({processed_residual_budget}) cannot exceed budget ({processed_budget})"
            )

        top_budget = processed_budget - processed_residual_budget

        # Calculate residual heap size
        if processed_residual_budget > 0:
            residual_heap_size = (size - top_budget) // processed_residual_budget
            if residual_clip > 0:
                residual_heap_size = min(residual_heap_size, residual_clip)
        else:
            residual_heap_size = 0

        return processed_budget, top_budget, processed_residual_budget, residual_heap_size

    @classmethod
    def compress_node(
        cls,
        node: "TreeNode",
        kv_pool: MHATokenToKVPool,
        token_allocator: BaseTokenToKVPoolAllocator,
        budget: int | float,
        residual_budget: int | float = 0,
        residual_clip: int = 0,
    ):
        """Compress a tree node's KV cache using Triton kernels.

        Args:
            node: The tree node to compress
            kv_pool: The KV cache pool containing K, V, and score buffers
            token_allocator: Allocator for KV cache locations
            budget: Number or fraction of tokens to keep after compression
            residual_budget: Budget for residual tokens (default: 0)
            residual_clip: Maximum size for residual heap (default: 0)
        """
        value = node.value
        size = len(value)

        # Process and validate budget parameters (inline to avoid circular import)
        processed_budget, top_budget, residual_budget_processed, residual_heap_size = (
            cls._process_budget_parameters(
                size, budget, residual_budget, residual_clip
            )
        )

        # Allocate new cache locations
        out_cache_loc = token_allocator.alloc(processed_budget)

        # Compress each layer using Triton kernels
        for layer_idx in range(kv_pool.layer_num):
            cls.compress_layer_triton(
                layer_idx=layer_idx,
                kv_pool=kv_pool,
                value=value,
                out_cache_loc=out_cache_loc,
                top_budget=top_budget,
                residual_budget=residual_budget_processed,
                residual_heap_size=residual_heap_size,
            )

        # Finalize: free old cache and update node
        token_allocator.free(value)
        node.value = out_cache_loc

    @staticmethod
    def compress_layer_triton(
        layer_idx: int,
        kv_pool: MHATokenToKVPool,
        value: torch.Tensor,
        out_cache_loc: torch.Tensor,
        top_budget: int,
        residual_budget: int,
        residual_heap_size: int,
    ):
        """Compress KV cache for a single layer using Triton kernels.

        This method uses fused Triton kernels for:
        1. Per-head top-k selection
        2. Residual token selection and sorting
        3. KV cache compression and averaging

        Args:
            layer_idx: The index of the layer to compress
            kv_pool: The KV cache pool
            value: Original cache locations (shape: [N])
            out_cache_loc: New cache locations (shape: [top_budget + residual_budget])
            top_budget: Number of high-score tokens to keep directly per head
            residual_budget: Number of compressed tokens from remaining tokens per head
            residual_heap_size: Size of each group for averaging (tokens per residual)
        """
        N = len(value)
        device = value.device

        # Get dimensions
        scores = kv_pool.s_buffer[layer_idx][value, :, 0]  # [N, head_num]
        head_num = scores.shape[1]
        head_dim = kv_pool.k_buffer[layer_idx].shape[-1]

        # Allocate output tensors for indices
        top_indices = torch.empty((head_num, top_budget), dtype=torch.int32, device=device)

        residual_source_count = residual_budget * residual_heap_size
        residual_indices = torch.empty(
            (head_num, residual_source_count) if residual_budget > 0 else (head_num, 1),
            dtype=torch.int32,
            device=device
        )

        # Step 1: Select top indices per head (kernel launch)
        grid = (head_num,)
        _select_top_indices_kernel[grid](
            scores,
            top_indices,
            N=N,
            head_num=head_num,
            top_budget=top_budget,
            scores_stride_n=scores.stride(0),
            scores_stride_h=scores.stride(1),
            top_indices_stride_h=top_indices.stride(0),
            top_indices_stride_k=top_indices.stride(1),
        )

        # Step 2: Select residual indices per head (if needed)
        if residual_budget > 0 and residual_heap_size > 0:
            _select_residual_indices_kernel[grid](
                scores,
                top_indices,
                residual_indices,
                N=N,
                head_num=head_num,
                top_budget=top_budget,
                residual_source_count=residual_source_count,
                scores_stride_n=scores.stride(0),
                scores_stride_h=scores.stride(1),
                top_indices_stride_h=top_indices.stride(0),
                top_indices_stride_k=top_indices.stride(1),
                residual_indices_stride_h=residual_indices.stride(0),
                residual_indices_stride_k=residual_indices.stride(1),
            )

        # Step 3: Compress KV cache (kernel launch)
        out_budget = top_budget + residual_budget
        BLOCK_SIZE = triton.next_power_of_2(min(head_dim, 128))

        grid = (out_budget, head_num)

        k_buffer = kv_pool.k_buffer[layer_idx]
        v_buffer = kv_pool.v_buffer[layer_idx]

        _compress_kv_kernel[grid](
            k_buffer,
            v_buffer,
            top_indices,
            residual_indices,
            value,
            k_buffer,  # Write in-place
            v_buffer,  # Write in-place
            out_cache_loc,
            N=N,
            head_num=head_num,
            head_dim=head_dim,
            top_budget=top_budget,
            residual_budget=residual_budget,
            residual_heap_size=residual_heap_size,
            out_budget=out_budget,
            k_in_stride_n=k_buffer.stride(0),
            k_in_stride_h=k_buffer.stride(1),
            k_in_stride_d=k_buffer.stride(2),
            v_in_stride_n=v_buffer.stride(0),
            v_in_stride_h=v_buffer.stride(1),
            v_in_stride_d=v_buffer.stride(2),
            k_out_stride_n=k_buffer.stride(0),
            k_out_stride_h=k_buffer.stride(1),
            k_out_stride_d=k_buffer.stride(2),
            v_out_stride_n=v_buffer.stride(0),
            v_out_stride_h=v_buffer.stride(1),
            v_out_stride_d=v_buffer.stride(2),
            top_indices_stride_h=top_indices.stride(0),
            top_indices_stride_k=top_indices.stride(1),
            residual_indices_stride_h=residual_indices.stride(0),
            residual_indices_stride_k=residual_indices.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reset buffers: clear old locations and initialize new locations
        kv_pool.s_buffer[layer_idx][value] = 0
        kv_pool.w_buffer[layer_idx][value] = -2
        kv_pool.w_buffer[layer_idx][out_cache_loc[:out_budget]] = 0
