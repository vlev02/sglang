"""
Base classes for KV Cache Compression

This module contains the fundamental abstract and pre-allocating base classes
for node compression. These are kept separate to avoid circular dependencies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool


class BaseNodeCompressor(ABC):
    """
    Abstract base class for node compression strategies.
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
        **kwargs
    ):
        if budget <= 0:
            raise ValueError(f"Invalid budget value: {budget=}")
        if page_size <= 0:
            raise ValueError(f"Invalid page_size value: {page_size=}")
        if residual_budget < 0:
            raise ValueError(f"Invalid residual_budget value: {residual_budget=}")
        if residual_clip < 0:
            raise ValueError(f"Invalid residual_clip value: {residual_clip=}")

        self.page_size = page_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        size = page_size
        if budget < 1:
            budget = int(size * budget)
        self.processed_budget = int(min(max(1, budget), size))

        if residual_budget < 1:
            residual_budget = int(self.processed_budget * residual_budget)
        self.processed_residual_budget = int(max(0, min(residual_budget, self.processed_budget)))

        if self.processed_residual_budget > self.processed_budget:
            raise ValueError(
                f"residual_budget ({self.processed_residual_budget}) cannot exceed budget ({self.processed_budget})"
            )

        self.top_budget = int(self.processed_budget - self.processed_residual_budget)

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

        if self.top_budget + self.processed_residual_budget != self.processed_budget:
            self.top_budget = self.processed_budget - self.processed_residual_budget

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
        size = len(value)
        if size == 0:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        if size % self.page_size:
            raise ValueError(
                f"Input value length ({size}) does not match expected page_size ({self.page_size}). "
                f"All nodes should have exactly page_size tokens."
            )

        if self.processed_budget == 0:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

        if self.processed_budget >= size:
            return torch.empty(0, dtype=value.dtype, device=value.device), value

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
        pass


class PreallocatingNodeCompressor(BaseNodeCompressor):
    """
    An optimized, stateful node compressor that pre-allocates and reuses GPU memory.
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
            device=device
        )
        
        if self.top_budget > 0:
            self.top_k_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )
            self.top_v_buffer = torch.empty(
                (self.top_budget, head_num, head_dim), dtype=dtype, device=device
            )

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
                    residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.view(residual_budget, residual_heap_size, head_num)

                    final_residual_locs = node_value[residual_indices_relative]
                    head_expanded = head_indices[None, None, :].expand(residual_budget, residual_heap_size, -1)

                    self.residual_k_buffer.copy_(k_buffer[final_residual_locs, head_expanded, :])
                    self.residual_v_buffer.copy_(v_buffer[final_residual_locs, head_expanded, :])

                    k_residual = self.residual_k_buffer.mean(dim=1)
                    v_residual = self.residual_v_buffer.sum(dim=1)

                    out_loc_res = node_out_cache_loc[top_budget:actual_budget, None]
                    head_range_scatter = head_indices[None, :].expand(residual_budget, -1)
                    k_buffer[out_loc_res, head_range_scatter, :] = k_residual
                    v_buffer[out_loc_res, head_range_scatter, :] = v_residual


class OptimizedNodeCompressor(BaseNodeCompressor):
    """
    Optimized node compressor with explicit memory cleanup.
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

                k_buffer[out_loc_range, head_range_expanded, :] = k_buffer[final_top_locs, head_range_expanded, :]
                v_buffer[out_loc_range, head_range_expanded, :] = v_buffer[final_top_locs, head_range_expanded, :]

                del top_indices_relative, final_top_locs, head_range_expanded, out_loc_range

            if residual_budget > 0 and residual_heap_size > 0:
                num_residual_source = residual_budget * residual_heap_size
                if N - top_budget >= num_residual_source:
                    residual_indices_relative = sorted_indices[top_budget:top_budget + num_residual_source, :]
                    residual_indices_relative, _ = torch.sort(residual_indices_relative, dim=0)
                    residual_indices_relative = residual_indices_relative.reshape(residual_budget, residual_heap_size, head_num)

                    final_residual_locs = node_value[residual_indices_relative]
                    head_expanded = head_indices[None, None, :].expand(residual_budget, residual_heap_size, -1)

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

        if device.type == 'cuda' and layer_idx % 4 == 3:
            torch.cuda.empty_cache()

__all__ = [
    "BaseNodeCompressor",
    "PreallocatingNodeCompressor",
    "OptimizedNodeCompressor",
]
