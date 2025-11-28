"""
Node Cache Compressor for KV Cache Compression

This module provides batch compression of radix tree nodes during decode phase.
It collects eligible nodes, allocates compressed space, performs compression,
and updates all metadata in a batched manner for efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.batch_compressor_base import BaseBatchCompressor, BatchCompressionTask

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class DecodeCompressionTask(BatchCompressionTask):
    """
    Extended compression task for decode-stage compression.

    Inherits from BatchCompressionTask and adds decode-specific metadata.

    Attributes:
        reqs: List of all requests with compressed nodes
        req_node_counts: Number of nodes compressed per request
        req_start_indices: Start index in node_li for each request
        req_prefix_start_indices: Start position in token pool for each compressed node chain
    """
    reqs: List["Req"] = None
    req_node_counts: List[int] = None
    req_start_indices: List[int] = None
    req_prefix_start_indices: List[int] = None

    def __post_init__(self):
        if self.req_node_counts is None:
            self.req_node_counts = []
        if self.req_start_indices is None:
            self.req_start_indices = []
        if self.req_prefix_start_indices is None:
            self.req_prefix_start_indices = []


class NodeCacheCompressor(BaseBatchCompressor):
    """
    Batched node cache compressor for efficient KV cache compression.

    This class collects all eligible nodes from a decode batch, allocates
    compressed space in batch, performs compression using GPU kernels,
    and updates all metadata (node values, tree stats, request indices)
    in a coordinated manner.

    Usage:
        compressor = NodeCacheCompressor(tree_cache, model_config)
        compressor.compress_batch_nodes(batch)
    """

    def __init__(
        self,
        tree_cache: "RadixCache",
        q_win_size: int,
        model_config,
        compress_stages: str = "0000",
    ):
        """
        Initialize the node cache compressor.

        Args:
            tree_cache: The radix cache tree
            q_win_size: Query window size for compression threshold
            model_config: Model configuration (for num_hidden_layers)
            compress_stages: 4-character string controlling compression at different stages
                           [during_prefill, after_prefill, during_decode, after_decode]
        """
        # Initialize base class
        super().__init__(tree_cache, model_config)

        # Decode-specific attributes
        self.q_win_size = q_win_size
        self.compression_tail_budget = tree_cache.compression_tail_budget
        self.compress_stages = compress_stages

    def compress_batch_nodes(self, batch: "ScheduleBatch") -> int:
        """
        Main entry point: compress all eligible nodes in a batch.

        Steps:
        1. Collect all compress-ready nodes from batch requests (explore ancestors)
        2. Allocate compressed indices for all nodes using continuous tensor
        3. Submit all compression tasks to GPU at once using batch tensors
        4. Update all metadata (nodes, tree stats, request indices)

        Args:
            batch: The decode batch that just completed

        Returns:
            Number of nodes successfully compressed
        """
        # Step 1: Collect compress-ready nodes (explore ancestor chains)
        task = self._collect_compression_tasks(batch)

        if task is None or len(task.node_li) == 0:
            return 0

        logger.debug(f"Collected {len(task.node_li)} compression nodes from {len(task.reqs)} requests")

        # Step 2: Allocate compressed indices (shared implementation from base class)
        if not self._allocate_compressed_indices(task):
            logger.warning("Failed to allocate compressed indices")
            return 0

        logger.debug(f"Successfully allocated continuous indices for {len(task.node_li)} nodes")

        # Step 3: Compress layers (shared implementation from base class)
        self._batch_compress_layers(task)

        logger.debug(f"Completed GPU compression for {len(task.node_li)} nodes")

        # Step 4: Update metadata (decode-specific implementation)
        tokens_saved = self._update_metadata(task)

        # Step 5: Store CompressionTask in batch for token pool update in prepare_for_decode
        batch.compression_task = task

        logger.info(
            f"Batch compression complete: {len(task.node_li)} nodes compressed "
            f"across {len(task.reqs)} requests, saved ~{tokens_saved} tokens"
        )

        return len(task.node_li)

    def _collect_compression_tasks(self, batch: "ScheduleBatch") -> Optional[DecodeCompressionTask]:
        """
        Step 1: Collect all compress-ready nodes from batch requests, exploring ancestor chains.

        For each request that meets the threshold, explores the path from last_node up to the root,
        collecting all nodes that are safe to compress. Builds continuous tensors for efficient
        batch processing.

        A node is ready for compression if:
        - Request is not retracted or finished
        - Request has generated exactly q_win_size new tokens (new_id_len == q_win_size)
        - Node id_len <= boundary_id_len (total_id_len - q_win_size)
        - Node size equals radix_block_size
        - Node lock_ref == 1 (exclusively held)

        Args:
            batch: The decode batch

        Returns:
            Single CompressionTask with batch-level data, or None if no nodes to compress
        """
        reqs = []
        node_li = []
        original_indices_list = []
        req_node_counts = []
        req_start_indices = []
        req_prefix_start_indices = []

        for req in batch.reqs:
            if req.is_retracted or req.finished():
                continue

            # Calculate new generated length (excluding the current token being generated)
            new_id_len = len(req.output_ids) - 1
            # Calculate boundary_id_len: total length minus compression tail budget
            total_id_len = len(req.origin_input_ids) + new_id_len
            boundary_id_len = total_id_len - self.compression_tail_budget

            # Check threshold: compress only when exactly at q_win_size
            if self.compress_stages[1] == '1' and new_id_len == self.q_win_size:
                pass
                
            elif self.compress_stages[2] == '1' and new_id_len > self.q_win_size and (total_id_len % self.radix_block_size == 0):
                pass
            else:
                continue

            # Explore ancestor chain from last_node up to root
            compress_nodes = self._collect_ancestor_nodes(req, boundary_id_len)

            if not compress_nodes:
                continue

            # Record request-level info
            reqs.append(req)
            req_start_indices.append(len(node_li))
            req_node_counts.append(len(compress_nodes))

            # Calculate prefix start index for this req's compressed chain
            # This is where the first compressed node starts in req.prefix_indices
            if compress_nodes:
                prefix_start = self._calculate_node_start_index(compress_nodes[0])
                req_prefix_start_indices.append(prefix_start)
            else:
                req_prefix_start_indices.append(0)

            # Collect nodes and build continuous tensor
            for node in compress_nodes:
                node_li.append(node)

                original_indices_list.append(node.value)

        if not node_li:
            return None

        # Build single continuous tensor for all original indices
        original_indices = torch.concat(original_indices_list, dim=0).to(self.device)

        task = DecodeCompressionTask(
            node_li=node_li,
            original_indices=original_indices,
            radix_block_size=self.radix_block_size,
            compressed_size=self.compressed_size,
            reqs=reqs,
            req_node_counts=req_node_counts,
            req_start_indices=req_start_indices,
            req_prefix_start_indices=req_prefix_start_indices,
        )
        return task

    def _collect_ancestor_nodes(self, req: "Req", boundary_id_len: int) -> List["TreeNode"]:
        """
        Collect all compressible ancestor nodes from last_node up the tree.

        Walks from req.last_node toward the root, collecting nodes that are safe to compress.
        Stops when reaching a node with id_len > boundary_id_len.
        Returns nodes in order from oldest ancestor to most recent (last_node last).

        Args:
            req: The request to explore
            boundary_id_len: Maximum id_len to consider for compression

        Returns:
            List of nodes to compress, ordered from ancestor to descendant
        """
        if req.last_node is None or req.last_node == self.tree_cache.root_node:
            return []

        compress_nodes = []
        current_node = req.last_node

        # Walk up the tree, collecting compressible nodes
        while current_node is not None and current_node != self.tree_cache.root_node:
            # Skip nodes beyond boundary_id_len (stop early since ancestors have larger id_len)
            if current_node.id_len > boundary_id_len:
                current_node = current_node.parent
                continue
            if self._is_node_safe_to_compress_direct(current_node):
                compress_nodes.append(current_node)

            current_node = current_node.parent

        # Reverse to get ancestor-to-descendant order
        compress_nodes.reverse()

        return compress_nodes

    def _is_node_safe_to_compress_direct(self, node: "TreeNode") -> bool:
        """
        Check if a node is safe to compress directly.

        Safety conditions:
        - Node size equals radix_block_size (aligned block)
        - Node is not locked by other requests (lock_ref == 1)

        Args:
            node: The tree node to check

        Returns:
            True if safe to compress, False otherwise
        """
        # Check node size matches radix_block_size (node.value is a tensor)
        node_size = node.value.size(0) if isinstance(node.value, torch.Tensor) else len(node.value)

        if node_size != self.radix_block_size:
            return False

        # Check lock reference (only compress if exclusively held by one request)
        if hasattr(node, 'lock_ref') and node.lock_ref != 1:
            return False

        return True

    def _calculate_node_start_index(self, node: "TreeNode") -> int:
        """
        Calculate where a node starts in the prefix_indices array.

        Walks up the tree from the node to the root, summing parent node sizes.

        Args:
            node: The tree node

        Returns:
            Starting index of this node in prefix_indices
        """
        # node_start_idx = 0
        # current_node = node

        # while current_node.parent is not None and current_node.parent != self.tree_cache.root_node:
        #     node_start_idx += len(current_node.parent.value)
        #     current_node = current_node.parent

        # return node_start_idx
        return node.idx_len - node.value.size(0)

    def _update_metadata(self, task: DecodeCompressionTask) -> int:
        """
        Step 4: Update all metadata after compression using batch tensor operations.

        Updates:
        - node.value: Replace with compressed indices for all nodes
        - node._id_len_ and ._idx_len_: Updated via TreeNode._update_lens()
        - tree_cache.protected_size_: Adjust for size change (batch update)
        - req.prefix_indices: Efficiently update using continuous tensors and tracked positions

        Args:
            task: The batch compression task

        Returns:
            Total number of tokens saved
        """
        if task.compressed_indices is None:
            return 0

        offset_compressed = 0
        total_node_count = len(task.node_li)

        # First pass: Update all node values
        for node_idx, node in enumerate(task.node_li):
            # Extract compressed indices for this node
            compressed_slice = task.compressed_indices[
                offset_compressed : offset_compressed + self.compressed_size
            ]

            # Update tree node value (keep as tensor for efficiency)
            node.value = compressed_slice.clone()

            offset_compressed += self.compressed_size

        # Update protected_size_ once for all nodes
        # These nodes are referenced by requests, so they're in protected_size_, not evictable_size_
        total_saved = total_node_count * (self.radix_block_size - self.compressed_size)
        self.tree_cache.protected_size_ -= total_saved
        # Free old KV cache locations once for all nodes
        self.token_to_kv_pool_allocator.free(task.original_indices)

        # Second pass: Update each request's metadata
        for req_idx, req in enumerate(task.reqs):
            # Update req.prefix_indices
            # self._update_request_prefix_indices_batch(task, req_idx)

            # Update length info from req's last_node and propagate to children
            for node_idx in task.req_start_indices:
                # Recursively update all children's length info
                self._update_children_lens(task.node_li[node_idx])

        return total_saved

    def _update_children_lens(self, node: "TreeNode"):
        """
        Recursively update _id_len_ and _idx_len_ for all children of a node.

        Args:
            node: The parent node whose children need updating
        """
        node._update_lens()
        for child in node.children.values():
            self._update_children_lens(child)

    def _update_request_prefix_indices_batch(self, task: CompressionTask, req_idx: int):
        """
        Efficiently update req.prefix_indices using continuous tensors for a request.

        Replaces the portion of prefix_indices corresponding to all compressed nodes
        in this request with the new compressed indices from the batch tensor.

        Args:
            task: The batch compression task
            req_idx: Index of the request in task.reqs
        """
        req = task.reqs[req_idx]

        # Get the node range for this request
        start_node_idx = task.req_start_indices[req_idx]
        node_count = task.req_node_counts[req_idx]

        # Calculate total size of original nodes for this request
        # All original nodes have size radix_block_size
        original_total_size = node_count * self.radix_block_size

        # Calculate total size of compressed nodes
        compressed_total_size = node_count * self.compressed_size

        # Get the starting position in req.prefix_indices
        prefix_start = task.req_prefix_start_indices[req_idx]

        # Extract compressed indices for all nodes of this request from continuous tensor
        compressed_offset = start_node_idx * self.compressed_size
        compressed_slice = task.compressed_indices[
            compressed_offset : compressed_offset + compressed_total_size
        ]

        # Build new prefix_indices
        # [prefix before compressed nodes] + [all compressed nodes] + [suffix after compressed nodes]
        req.prefix_indices = torch.cat([
            req.prefix_indices[:prefix_start],                              # Before
            compressed_slice,                                                # All compressed nodes
            req.prefix_indices[prefix_start + original_total_size:]        # After
        ])

        logger.debug(
            f"Updated req {req.rid} prefix_indices: "
            f"len {len(req.prefix_indices)} "
            f"(compressed {node_count} nodes at {prefix_start}, {original_total_size} → {compressed_total_size})"
        )
