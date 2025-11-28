"""
Batched Node Compressor for High-Efficiency KV Cache Compression

This module provides a batched implementation of the node compressor, which
processes all nodes in a single operation to maximize GPU utilization.
"""

import torch
from typing import TYPE_CHECKING

from sglang.srt.mem_cache.base_compressor import PreallocatingNodeCompressor

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class BatchedNodeCompressor(PreallocatingNodeCompressor):
    """
    A node compressor that processes all nodes in a batch in a single,
    vectorized operation, eliminating Python loops for maximum efficiency.

    This implementation overrides the `compress_layer` method to perform
    batched gather, sort, reduction, and scatter operations.
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
        """
        Compresses a layer for a batch of nodes simultaneously.
        """
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

        # --- Batched Score Gathering and Sorting ---
        # scores: [B, N, H]
        scores = s_buffer[node_values, :, 0]
        # sorted_indices: [B, N, H]
        _, sorted_indices = torch.sort(scores, dim=1, descending=True)

        # --- Batched Top-K Processing ---
        if top_budget > 0:
            # Get relative indices for top-k tokens: [B, T, H]
            top_indices_relative = sorted_indices[:, :top_budget, :]
            
            # Gather the absolute token locations from `node_values`: [B, T, H]
            final_top_locs = torch.gather(node_values.unsqueeze(2).expand(-1, -1, head_num), 1, top_indices_relative)

            # To gather from k_buffer[pool_size, H, D], we align dimensions and flatten B and H.
            # Permute to [B, H, T] then flatten to [B*H, T]
            final_top_locs_flat = final_top_locs.permute(0, 2, 1).reshape(num_nodes * head_num, top_budget)
            
            # Create a head index tensor that matches final_top_locs_flat: [B*H]
            head_indices_flat = torch.arange(head_num, device=self.device).repeat(num_nodes)
            
            # Gather K/V values. The source is k_buffer indexed by [token_loc, head_idx, :]
            # The indexing becomes: k_buffer[final_top_locs_flat.T, head_indices_flat, :]
            # This results in a shape of [T, B*H, D]
            top_k_gathered = k_buffer[final_top_locs_flat.T, head_indices_flat, :]
            top_v_gathered = v_buffer[final_top_locs_flat.T, head_indices_flat, :]

            # Scatter the gathered values to the output locations.
            # out_locs_top: [B, T] -> [B, 1, T] -> [B, H, T]
            out_locs_top = node_out_cache_locs[:, :top_budget].unsqueeze(1).expand(-1, head_num, -1)
            # out_locs_top_flat: [B*H, T]
            out_locs_top_flat = out_locs_top.reshape(num_nodes * head_num, top_budget)

            # Scatter operation: k_buffer[out_locs_top_flat.T, head_indices_flat, :] = top_k_gathered
            k_buffer[out_locs_top_flat.T, head_indices_flat, :] = top_k_gathered
            v_buffer[out_locs_top_flat.T, head_indices_flat, :] = top_v_gathered

        # --- Batched Residual Processing ---
        if residual_budget > 0 and residual_heap_size > 0:
            num_residual_source = residual_budget * residual_heap_size
            if self.page_size - top_budget >= num_residual_source:
                # Get relative indices for residual tokens: [B, R*heap, H]
                residual_indices_relative = sorted_indices[:, top_budget : top_budget + num_residual_source, :]
                residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=1)
                # Reshape for grouping: [B, R, heap, H]
                residual_indices_relative = residual_indices_relative.view(num_nodes, residual_budget, residual_heap_size, head_num)

                # Gather absolute token locations: [B, R, heap, H]
                final_residual_locs = torch.gather(
                    node_values.view(num_nodes, 1, self.page_size, 1).expand(-1, residual_budget, -1, head_num),
                    2,
                    residual_indices_relative
                )

                # Flatten for gather: [B*H*R, heap]
                final_residual_locs_flat = final_residual_locs.permute(0, 3, 1, 2).reshape(num_nodes * head_num * residual_budget, residual_heap_size)
                
                # Create matching head indices: [B*H*R]
                head_indices_res_flat = torch.arange(head_num, device=self.device).repeat_interleave(residual_budget).repeat(num_nodes)

                # Gather residual K/V values: [heap, B*H*R, D]
                residual_k_gathered = k_buffer[final_residual_locs_flat.T, head_indices_res_flat, :]
                residual_v_gathered = v_buffer[final_residual_locs_flat.T, head_indices_res_flat, :]

                # Perform reduction
                # k_residual: [B*H*R, D]
                k_residual = residual_k_gathered.mean(dim=0)
                v_residual = residual_v_gathered.sum(dim=0)

                # Scatter the reduced values
                # out_locs_res: [B, R] -> [B, 1, R] -> [B, H, R]
                out_locs_res = node_out_cache_locs[:, top_budget:actual_budget].unsqueeze(1).expand(-1, head_num, -1)
                # out_locs_res_flat: [B*H, R] -> [B*H*R]
                out_locs_res_flat = out_locs_res.reshape(num_nodes * head_num, residual_budget).reshape(-1)

                # Scatter operation
                k_buffer[out_locs_res_flat, head_indices_res_flat, :] = k_residual
                v_buffer[out_locs_res_flat, head_indices_res_flat, :] = v_residual