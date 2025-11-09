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
from sglang.srt.mem_cache.batched_compressor import BatchedNodeCompressor
from sglang.srt.mem_cache.base_compressor import BaseNodeCompressor, PreallocatingNodeCompressor, OptimizedNodeCompressor
from sglang.srt.mem_cache.gpu_compression_optimizations import OptimizedCompressionPipeline
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.managers.schedule_batch import BaseTokenToKVPoolAllocator


class VanillaNodeCompressor(BaseNodeCompressor):
    """
    Original vanilla PyTorch implementation.
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
                top_budget + residual_budget, head_num, k_values.shape[2],
                dtype=k_values.dtype, device=device
            )
            combined_v = torch.zeros(
                top_budget + residual_budget, head_num, v_values.shape[2],
                dtype=v_values.dtype, device=device
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
                        residual_source_indices = sorted_indices[top_budget:top_budget + num_residual_source]
                        residual_source_indices, _ = torch.sort(residual_source_indices)
                        residual_source_indices = residual_source_indices.reshape(residual_budget, residual_heap_size)

                        residual_k = k_values[residual_source_indices, head_idx, :]
                        residual_v = v_values[residual_source_indices, head_idx, :]

                        combined_k[top_budget:, head_idx, :] = residual_k.mean(dim=1)
                        combined_v[top_budget:, head_idx, :] = residual_v.sum(dim=1)

            actual_budget = top_budget + residual_budget
            if actual_budget > 0:
                k_buffer[node_out_cache_loc[:actual_budget]] = combined_k[:actual_budget]
                v_buffer[node_out_cache_loc[:actual_budget]] = combined_v[:actual_budget]


# from sglang.srt.mem_cache.gpu_compression_optimizations import OptimizedCompressionPipeline


class UltraOptimizedNodeCompressor(BatchedNodeCompressor):
    pass


# Default compressor to use
NodeCompressor = UltraOptimizedNodeCompressor

__all__ = [
    "BaseNodeCompressor",
    "VanillaNodeCompressor",
    "OptimizedNodeCompressor",
    "PreallocatingNodeCompressor",
    "UltraOptimizedNodeCompressor",
    "NodeCompressor",
]
        
        