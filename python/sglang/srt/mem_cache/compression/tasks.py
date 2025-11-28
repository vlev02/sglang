"""
Compression Task Data Structures

This module contains all dataclass definitions for compression tasks,
keeping data structures separate from logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.managers.schedule_batch import Req


@dataclass
class CompressionTask:
    """
    Base compression task for batched node compression.

    Attributes:
        node_list: List of tree nodes to compress
        original_indices: Continuous tensor of original KV cache indices
        compressed_indices: Continuous tensor for compressed KV cache indices (allocated later)
        page_size: Size of each original node (radix block size)
        compressed_size: Size of each compressed node (compression budget)
    """

    node_list: List["TreeNode"]
    original_indices: torch.Tensor
    compressed_indices: Optional[torch.Tensor] = None
    page_size: int = 0
    compressed_size: int = 0


@dataclass
class InsertionCompressionTask(CompressionTask):
    """
    Compression task for cache insertion stages.

    Used during:
    - Prefill stage (compress_stages[0])
    - After request finish (compress_stages[3])

    Tracks newly inserted nodes that need compression.
    """

    pass  # Inherits all fields from CompressionTask


@dataclass
class DecodeCompressionTask(CompressionTask):
    """
    Compression task for decode-stage batch compression.

    Used during:
    - After prefill (compress_stages[1])
    - During decode (compress_stages[2])

    Tracks additional request-level metadata for efficient batch updates.

    Attributes:
        req_list: List of requests with compressed nodes
        req_node_counts: Number of nodes compressed per request
        req_start_indices: Start index in node_list for each request
        req_prefix_start_indices: Start position in token pool for each compressed chain
    """

    req_list: List["Req"] = field(default_factory=list)
    req_node_counts: List[int] = field(default_factory=list)
    req_start_indices: List[int] = field(default_factory=list)
    req_prefix_start_indices: List[int] = field(default_factory=list)


__all__ = [
    "CompressionTask",
    "InsertionCompressionTask",
    "DecodeCompressionTask",
]
