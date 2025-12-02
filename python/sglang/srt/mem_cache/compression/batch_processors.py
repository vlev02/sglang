"""
Batch Compression Processors

This module contains batch-level compression coordinators that orchestrate
multi-node compression operations across different pipeline stages.

Architecture:
    BaseBatchProcessor (ABC)
    ├── InsertionBatchProcessor   - For insertion/prefill stages
    └── DecodeBatchProcessor      - For decode stages
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.compression.tasks import (
    CompressionTask,
    InsertionCompressionTask,
    DecodeCompressionTask,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class BaseBatchProcessor(ABC):
    """
    Abstract base class for batch compression processors.

    Defines the standard 4-step compression pipeline:
    1. Collect nodes (_collect_tasks)
    2. Allocate compressed indices (_allocate_indices)
    3. Compress layers (_compress_layers) - GPU operation
    4. Update metadata (_update_metadata)

    Steps 2 and 3 are shared; steps 1 and 4 are stage-specific.
    """

    def __init__(
        self,
        tree_cache: "RadixCache",
        model_config,
    ):
        """
        Initialize batch processor.

        Args:
            tree_cache: The radix cache tree
            model_config: Model configuration (for num_hidden_layers)
        """
        # Validate
        if not hasattr(tree_cache, "compressor") or tree_cache.compressor is None:
            raise ValueError("RadixCache must have a compressor configured")

        # Store references
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
        self.device = tree_cache.device
        self.model_config = model_config
        self.page_size = tree_cache.page_size

        # Reuse existing single-node compressor
        self.compression_engine = tree_cache.compressor
        self.compressed_size = self.compression_engine.processed_budget

    @abstractmethod
    def _collect_tasks(self, *args, **kwargs) -> Optional[CompressionTask]:
        """
        Step 1: Collect nodes to compress (stage-specific).

        Must build a CompressionTask with continuous tensors.

        Returns:
            CompressionTask with collected nodes, or None if nothing to compress
        """
        pass

    def _allocate_indices(self, task: CompressionTask) -> bool:
        """
        Step 2: Allocate compressed KV cache indices as a continuous tensor.

        This is shared across all compression stages.

        Args:
            task: The compression task

        Returns:
            True if allocation succeeds, False otherwise
        """
        if len(task.node_list) == 0:
            return False

        try:
            # Allocate continuous tensor
            required_size = self.compressed_size * len(task.node_list)
            compressed_indices = self.token_to_kv_pool_allocator.alloc(required_size)

            if compressed_indices is None:
                logger.warning(f"Failed to allocate {required_size} compressed indices")
                return False

            task.compressed_indices = compressed_indices
            return True

        except Exception as e:
            logger.warning(f"Failed to allocate compressed indices: {e}")
            return False

    def _compress_layers(self, task: CompressionTask):
        """
        Step 3: Compress all nodes for all layers using GPU kernels.

        This is shared across all compression stages.

        Args:
            task: The compression task with continuous tensors
        """
        num_layers = self.model_config.num_hidden_layers
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()

        try:
            # Compress all layers
            for layer_idx in range(num_layers):
                self.compression_engine.compress_layer(
                    layer_idx=layer_idx,
                    kv_pool=kv_pool,
                    value=task.original_indices,  # Entire continuous tensor
                    out_cache_loc=task.compressed_indices,  # Entire continuous tensor
                    top_budget=self.compression_engine.top_budget,
                    residual_budget=self.compression_engine.processed_residual_budget,
                    residual_heap_size=self.compression_engine.residual_heap_size,
                )

            logger.debug(f"Compressed {len(task.node_list)} nodes across {num_layers} layers")

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            task.compressed_indices = None

    @abstractmethod
    def _update_metadata(self, task: CompressionTask) -> int:
        """
        Step 4: Update metadata after compression (stage-specific).

        Must update tree nodes, request indices, and other metadata.

        Args:
            task: The compression task

        Returns:
            Number of tokens saved by compression
        """
        pass

    def _is_node_compressible(self, node: "TreeNode") -> bool:
        """
        Check if a node is safe to compress.

        Args:
            node: The tree node to check

        Returns:
            True if node can be compressed, False otherwise
        """
        if node.value is None:
            return False

        # Check node size matches page_size
        node_size = node.value.size(0) if isinstance(node.value, torch.Tensor) else len(node.value)
        return node_size == self.page_size


class InsertionBatchProcessor(BaseBatchProcessor):
    """
    Batch processor for cache insertion stages.

    Used by:
    - cache_unfinished_req() (compress_stages[0] - during prefill)
    - cache_finished_req() (compress_stages[3] - after finish)

    Collects newly inserted nodes and compresses them in batch.

    Usage:
        processor = InsertionBatchProcessor(tree_cache, model_config)
        processor.begin_batch()
        # ... insert operations that call mark_node_for_compression() ...
        processor.compress_batch()
    """

    def __init__(self, tree_cache: "RadixCache", model_config):
        super().__init__(tree_cache, model_config)
        self.compression_tail_budget = tree_cache.compression_tail_budget

        # Accumulator for pending nodes
        self._pending_nodes: List["TreeNode"] = []

    def begin_batch(self):
        """Start a batch insertion operation."""
        self._pending_nodes = []

    def mark_node_for_compression(self, node: "TreeNode"):
        """
        Mark a newly inserted node for batched compression.

        Called from _insert_helper() instead of immediate compression.

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
        task = self._collect_tasks()

        if task is None or len(task.node_list) == 0:
            return 0

        logger.debug(f"Collected {len(task.node_list)} nodes for insertion compression")

        if not self._allocate_indices(task):
            logger.warning("Failed to allocate compressed indices for insertion")
            return 0

        self._compress_layers(task)
        self._update_metadata(task)

        # Clear pending nodes
        num_compressed = len(task.node_list)
        self._pending_nodes = []

        logger.info(f"Insertion compression complete: {num_compressed} nodes compressed")

        return num_compressed

    def _collect_tasks(self) -> Optional[InsertionCompressionTask]:
        """
        Step 1: Collect all pending nodes for compression.

        Filters pending nodes and builds continuous tensors for batch processing.

        Returns:
            InsertionCompressionTask or None if no nodes to compress
        """
        if not self._pending_nodes:
            return None

        # Filter compressible nodes
        compress_nodes = [
            node for node in self._pending_nodes if self._is_node_compressible(node)
        ]

        if not compress_nodes:
            logger.debug("No compressible nodes found in pending list")
            return None

        # Build continuous tensors
        original_indices_list = [node.value for node in compress_nodes]
        original_indices = torch.cat(original_indices_list, dim=0).to(self.device)

        return InsertionCompressionTask(
            node_list=compress_nodes,
            original_indices=original_indices,
            page_size=self.page_size,
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
            task: The compression task

        Returns:
            Number of tokens saved
        """
        if task.compressed_indices is None:
            return 0

        offset = 0
        total_saved = 0

        for node in task.node_list:
            # Extract compressed indices for this node
            compressed_slice = task.compressed_indices[offset : offset + self.compressed_size]

            # Save original for freeing
            original_value = node.value
            original_size = len(original_value)

            # Update node with compressed indices
            node.value = compressed_slice.clone()

            # Update tree cache size (new nodes are in evictable_size_)
            size_delta = original_size - self.compressed_size
            self.tree_cache.evictable_size_ -= size_delta
            total_saved += size_delta

            # Free original KV cache
            self.token_to_kv_pool_allocator.free(original_value)

            # Update node lengths
            node._update_lens()

            offset += self.compressed_size

        logger.debug(f"Updated metadata for {len(task.node_list)} nodes, saved {total_saved} tokens")

        return total_saved


class DecodeBatchProcessor(BaseBatchProcessor):
    """
    Batch processor for decode-stage compression.

    Used by:
    - After prefill (compress_stages[1])
    - During decode (compress_stages[2])

    Collects eligible nodes from decode batch and compresses them.

    Usage:
        processor = DecodeBatchProcessor(tree_cache, q_win_size, model_config, compress_stages)
        processor.compress_batch(schedule_batch)
    """

    def __init__(
        self,
        tree_cache: "RadixCache",
        q_win_size: int,
        model_config,
        compress_stages: str = "0000",
    ):
        super().__init__(tree_cache, model_config)

        self.q_win_size = q_win_size
        self.compression_tail_budget = tree_cache.compression_tail_budget
        self.compress_stages = compress_stages

    def compress_batch(self, batch: "ScheduleBatch") -> int:
        """
        Compress all eligible nodes in a decode batch.

        Steps:
        1. Collect compress-ready nodes (explore ancestors)
        2. Allocate compressed indices
        3. Compress layers (GPU)
        4. Update metadata (nodes, tree stats, request indices)

        Args:
            batch: The decode batch

        Returns:
            Number of nodes successfully compressed
        """
        import time

        # Step 1: Collect nodes
        t_collect_start = time.time()
        task = self._collect_tasks(batch)
        t_collect_end = time.time()

        if task is None or len(task.node_list) == 0:
            logger.debug(f"[PERF_COMPRESS_DETAIL] collect:{(t_collect_end-t_collect_start)*1000:.2f}ms nodes:0 allocate:0ms compress:0ms update:0ms")
            return 0

        logger.debug(f"Collected {len(task.node_list)} nodes from {len(task.req_list)} requests")

        # Step 2: Allocate indices
        t_allocate_start = time.time()
        alloc_success = self._allocate_indices(task)
        t_allocate_end = time.time()

        if not alloc_success:
            logger.warning("Failed to allocate compressed indices")
            logger.debug(f"[PERF_COMPRESS_DETAIL] collect:{(t_collect_end-t_collect_start)*1000:.2f}ms nodes:{len(task.node_list)} allocate:{(t_allocate_end-t_allocate_start)*1000:.2f}ms(FAILED) compress:0ms update:0ms")
            return 0

        logger.debug(f"Allocated indices for {len(task.node_list)} nodes")

        # Step 3: Compress layers
        t_compress_start = time.time()
        self._compress_layers(task)
        t_compress_end = time.time()

        logger.debug(f"Completed GPU compression for {len(task.node_list)} nodes")

        # Step 4: Update metadata
        t_update_start = time.time()
        tokens_saved = self._update_metadata(task)
        t_update_end = time.time()

        # Step 5: Store task in batch for token pool update
        batch.compression_task = task

        logger.info(
            f"Batch compression complete: {len(task.node_list)} nodes compressed "
            f"across {len(task.req_list)} requests, saved ~{tokens_saved} tokens"
        )
        logger.debug(f"[PERF_COMPRESS_DETAIL] collect:{(t_collect_end-t_collect_start)*1000:.2f}ms nodes:{len(task.node_list)} allocate:{(t_allocate_end-t_allocate_start)*1000:.2f}ms compress:{(t_compress_end-t_compress_start)*1000:.2f}ms update:{(t_update_end-t_update_start)*1000:.2f}ms")

        return len(task.node_list)

    def compress_batch_nodes(self, batch: "ScheduleBatch") -> int:
        """
        Compatibility alias for compress_batch().

        This method exists for backward compatibility with existing code
        that calls compress_batch_nodes() instead of compress_batch().

        Args:
            batch: The decode batch

        Returns:
            Number of nodes successfully compressed
        """
        return self.compress_batch(batch)

    def _collect_tasks(self, batch: "ScheduleBatch") -> Optional[DecodeCompressionTask]:
        """
        Step 1: Collect compress-ready nodes from batch, exploring ancestor chains.

        A node is ready if:
        - Request is not retracted or finished
        - Request meets generation threshold
        - Node id_len <= boundary_id_len
        - Node size equals page_size
        - Node lock_ref == 1 (exclusively held)

        Args:
            batch: The decode batch

        Returns:
            DecodeCompressionTask or None
        """
        req_list = []
        node_list = []
        original_indices_list = []
        req_node_counts = []
        req_start_indices = []
        req_prefix_start_indices = []

        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue

            # Calculate lengths
            new_id_len = len(req.output_ids) - 1
            total_id_len = len(req.origin_input_ids) + new_id_len
            boundary_id_len = total_id_len - self.compression_tail_budget

            # Check threshold
            if self.compress_stages[1] == "1" and new_id_len == self.q_win_size:
                pass
            elif (
                self.compress_stages[2] == "1"
                and new_id_len > self.q_win_size
                and (total_id_len % self.page_size == 0)
            ):
                pass
            else:
                continue

            # Explore ancestor chain
            compress_nodes = self._collect_ancestor_nodes(req, boundary_id_len)

            if not compress_nodes:
                continue

            # Record request-level info
            req_list.append(req)
            req_start_indices.append(len(node_list))
            req_node_counts.append(len(compress_nodes))

            # Calculate prefix start index
            if compress_nodes:
                prefix_start = self._calculate_node_start_index(compress_nodes[0])
                req_prefix_start_indices.append(prefix_start)
            else:
                req_prefix_start_indices.append(0)

            # Collect nodes
            for node in compress_nodes:
                node_list.append(node)
                original_indices_list.append(node.value)

        if not node_list:
            return None

        # Build continuous tensor
        original_indices = torch.concat(original_indices_list, dim=0).to(self.device)

        return DecodeCompressionTask(
            node_list=node_list,
            original_indices=original_indices,
            page_size=self.page_size,
            compressed_size=self.compressed_size,
            req_list=req_list,
            req_node_counts=req_node_counts,
            req_start_indices=req_start_indices,
            req_prefix_start_indices=req_prefix_start_indices,
        )

    def _collect_ancestor_nodes(self, req, boundary_id_len: int) -> List["TreeNode"]:
        """
        Collect compressible ancestor nodes from last_node up the tree.

        Args:
            req: The request
            boundary_id_len: Maximum id_len to consider

        Returns:
            List of nodes ordered from ancestor to descendant
        """
        if req.last_node is None or req.last_node == self.tree_cache.root_node:
            return []

        compress_nodes = []
        current_node = req.last_node

        # Walk up the tree
        while current_node is not None and current_node != self.tree_cache.root_node:
            # Skip nodes beyond boundary
            if current_node.id_len > boundary_id_len:
                current_node = current_node.parent
                continue

            if self._is_node_safe_to_compress(current_node):
                compress_nodes.append(current_node)

            current_node = current_node.parent

        # Reverse to get ancestor-to-descendant order
        compress_nodes.reverse()

        return compress_nodes

    def _is_node_safe_to_compress(self, node: "TreeNode") -> bool:
        """
        Check if a node is safe to compress directly.

        Args:
            node: The tree node

        Returns:
            True if safe to compress
        """
        # Check node size
        node_size = node.value.size(0) if isinstance(node.value, torch.Tensor) else len(node.value)

        if node_size != self.page_size:
            return False

        # Check lock reference
        if hasattr(node, "lock_ref") and node.lock_ref != 1:
            return False

        return True

    def _calculate_node_start_index(self, node: "TreeNode") -> int:
        """
        Calculate where a node starts in prefix_indices.

        Args:
            node: The tree node

        Returns:
            Starting index
        """
        return node.idx_len - node.value.size(0)

    def _update_metadata(self, task: DecodeCompressionTask) -> int:
        """
        Step 4: Update all metadata after compression.

        Updates:
        - node.value: Replace with compressed indices
        - node lengths: Update via _update_lens()
        - tree_cache.protected_size_: Adjust for size change
        - Free original KV cache locations

        Args:
            task: The compression task

        Returns:
            Total tokens saved
        """
        if task.compressed_indices is None:
            return 0

        offset_compressed = 0

        # Update all node values
        for node in task.node_list:
            compressed_slice = task.compressed_indices[
                offset_compressed : offset_compressed + self.compressed_size
            ]

            # Update node value
            node.value = compressed_slice.clone()

            offset_compressed += self.compressed_size

        # Update protected_size_ once for all nodes
        total_saved = len(task.node_list) * (self.page_size - self.compressed_size)
        self.tree_cache.protected_size_ -= total_saved

        # Free old KV cache locations
        self.token_to_kv_pool_allocator.free(task.original_indices)

        # Update request metadata
        for req_idx, req in enumerate(task.req_list):
            # Update length info from req's nodes
            for node_idx in task.req_start_indices:
                # Recursively update children's lengths
                self._update_children_lens(task.node_list[node_idx])

        return total_saved

    def _update_children_lens(self, node: "TreeNode"):
        """
        Recursively update _id_len_ and _idx_len_ for all children.

        Args:
            node: The parent node
        """
        node._update_lens()
        for child in node.children.values():
            self._update_children_lens(child)


__all__ = [
    "BaseBatchProcessor",
    "InsertionBatchProcessor",
    "DecodeBatchProcessor",
]
