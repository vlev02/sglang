"""
Batched compressor for cache insertion stages.

This module provides batched compression for nodes created during tree insertion
operations (cache_unfinished_req and cache_finished_req stages).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.batch_compressor_base import BaseBatchCompressor, BatchCompressionTask

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class InsertionCompressionTask(BatchCompressionTask):
    """
    Compression task for insertion stages.

    Tracks new nodes created during insertion that need compression.
    Inherits base attributes from BatchCompressionTask.
    """
    pass  # No additional attributes needed for insertion stage


class InsertionCacheCompressor(BaseBatchCompressor):
    """
    Batched compressor for cache insertion operations.

    Used by:
    - cache_unfinished_req() (compress_stages[0] - during prefill)
    - cache_finished_req() (compress_stages[3] - after finish)

    Instead of compressing nodes immediately during _insert_helper(),
    this collects all new nodes and compresses them in batch.

    Usage:
        compressor = InsertionCacheCompressor(tree_cache, model_config)
        compressor.begin_batch()
        # ... insert operations that call mark_node_for_compression() ...
        compressor.compress_batch()
    """

    def __init__(self, tree_cache: "RadixCache", model_config):
        """
        Initialize the insertion cache compressor.

        Args:
            tree_cache: The radix cache tree
            model_config: Model configuration (for num_hidden_layers)
        """
        super().__init__(tree_cache, model_config)
        self.compression_tail_budget = tree_cache.compression_tail_budget

        # Accumulator for nodes pending compression
        self._pending_nodes: List["TreeNode"] = []

    def begin_batch(self):
        """
        Start a batch insertion operation.
        Clears the pending nodes accumulator.
        """
        self._pending_nodes = []

    def mark_node_for_compression(self, node: "TreeNode"):
        """
        Mark a newly inserted node for batched compression.

        Called from _insert_helper() instead of _compress_node().

        Args:
            node: Tree node to compress later
        """
        self._pending_nodes.append(node)

    def compress_batch(self) -> int:
        """
        Compress all pending nodes accumulated during batch insertion.

        Returns:
            Number of nodes successfully compressed
        """
        task = self._collect_compression_tasks()

        if task is None or len(task.node_li) == 0:
            return 0

        logger.debug(f"Collected {len(task.node_li)} nodes for insertion compression")

        if not self._allocate_compressed_indices(task):
            logger.warning("Failed to allocate compressed indices for insertion")
            return 0

        self._batch_compress_layers(task)
        self._update_metadata(task)

        # Clear pending nodes
        num_compressed = len(task.node_li)
        self._pending_nodes = []

        logger.info(f"Insertion compression complete: {num_compressed} nodes compressed")

        return num_compressed

    def _collect_compression_tasks(self) -> Optional[InsertionCompressionTask]:
        """
        Step 1: Collect all pending nodes for compression.

        Filters pending nodes to ensure they are safe to compress,
        then builds continuous tensors for batch processing.

        Returns:
            Task with continuous tensors for batch compression, or None if no nodes to compress
        """
        if not self._pending_nodes:
            return None

        # Filter nodes that are safe to compress
        compress_nodes = [
            node for node in self._pending_nodes
            if self._is_node_compressible(node)
        ]

        if not compress_nodes:
            logger.debug("No compressible nodes found in pending list")
            return None

        # Build continuous tensors from all node values
        original_indices_list = [node.value for node in compress_nodes]
        original_indices = torch.cat(original_indices_list, dim=0).to(self.device)

        return InsertionCompressionTask(
            node_li=compress_nodes,
            original_indices=original_indices,
            radix_block_size=self.radix_block_size,
            compressed_size=self.compressed_size,
        )

    def _update_metadata(self, task: InsertionCompressionTask) -> int:
        """
        Step 4: Update metadata after compression.

        Updates:
        - node.value: Replace with compressed indices
        - tree_cache.evictable_size_: Adjust for size change
        - Free original KV cache locations

        Args:
            task: The batch compression task

        Returns:
            Number of tokens saved
        """
        if task.compressed_indices is None:
            return 0

        offset = 0
        total_saved = 0

        for node in task.node_li:
            # Extract compressed indices for this node
            compressed_slice = task.compressed_indices[
                offset : offset + self.compressed_size
            ]

            # Save original for freeing and tracking
            original_value = node.value
            original_size = len(original_value)

            # Update node with compressed indices
            node.value = compressed_slice.clone()

            # Update tree cache size (these are new nodes, so they're in evictable_size_)
            size_delta = original_size - self.compressed_size
            self.tree_cache.evictable_size_ -= size_delta
            total_saved += size_delta

            # Free original KV cache
            self.token_to_kv_pool_allocator.free(original_value)

            # Update node lengths
            node._update_lens()

            offset += self.compressed_size

        logger.debug(f"Updated metadata for {len(task.node_li)} nodes, saved {total_saved} tokens")

        return total_saved
