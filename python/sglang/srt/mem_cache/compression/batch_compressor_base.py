"""
Abstract base class for batch compression operations.

This module provides the common infrastructure for batched KV cache compression
across different stages (insertion, decode, etc.). It defines the standard 4-step
compression pipeline and reuses the existing NodeCompressor.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator

logger = logging.getLogger(__name__)


@dataclass
class BatchCompressionTask:
    """
    Base class for compression task data.

    Attributes:
        node_li: List of all nodes to compress
        original_indices: Continuous tensor containing all original KV indices
        compressed_indices: Continuous tensor for all compressed KV indices
        radix_block_size: Size of each original node
        compressed_size: Size of each compressed node
    """
    node_li: List["TreeNode"]
    original_indices: torch.Tensor
    compressed_indices: Optional[torch.Tensor] = None
    radix_block_size: int = 0
    compressed_size: int = 0


class BaseBatchCompressor(ABC):
    """
    Abstract base class for batched node compression.

    Defines the standard 4-step compression pipeline:
    1. Collect nodes (_collect_compression_tasks)
    2. Allocate compressed indices (_allocate_compressed_indices)
    3. Compress layers (_batch_compress_layers) - GPU operation
    4. Update metadata (_update_metadata)

    Steps 2 and 3 are shared across all implementations.
    Steps 1 and 4 are stage-specific and must be implemented by subclasses.
    """

    def __init__(
        self,
        tree_cache: "RadixCache",
        model_config,
    ):
        """
        Initialize the batch compressor.

        Args:
            tree_cache: The radix cache tree
            model_config: Model configuration (for num_hidden_layers)
        """
        # Check if compressor is available
        if not hasattr(tree_cache, 'compressor') or tree_cache.compressor is None:
            raise ValueError("RadixCache must have a compressor configured")

        # Extract attributes from tree_cache
        self.tree_cache = tree_cache
        self.token_to_kv_pool_allocator = tree_cache.token_to_kv_pool_allocator
        self.device = tree_cache.device
        self.model_config = model_config
        self.radix_block_size = tree_cache.page_size

        # Reuse existing compressor
        self.compressor = tree_cache.compressor
        self.compressed_size = self.compressor.processed_budget

    @abstractmethod
    def _collect_compression_tasks(self, *args, **kwargs) -> Optional[BatchCompressionTask]:
        """
        Step 1: Collect nodes to compress (stage-specific).

        Must be implemented by subclasses to define what nodes to compress
        and build the BatchCompressionTask with continuous tensors.

        Returns:
            BatchCompressionTask with collected nodes, or None if nothing to compress
        """
        pass

    def _allocate_compressed_indices(self, task: BatchCompressionTask) -> bool:
        """
        Step 2: Allocate compressed KV cache indices as a continuous tensor.

        This is shared across all compression stages.

        Args:
            task: The batch compression task

        Returns:
            True if allocation succeeds, False otherwise
        """
        if len(task.node_li) == 0:
            return False

        try:
            # Allocate continuous tensor for all compressed indices
            required_ind_num = self.compressed_size * len(task.node_li)
            compressed_indices = self.token_to_kv_pool_allocator.alloc(required_ind_num)

            if compressed_indices is None:
                logger.warning(f"Failed to allocate {required_ind_num} compressed indices")
                return False

            task.compressed_indices = compressed_indices
            return True

        except Exception as e:
            logger.warning(f"Failed to allocate compressed indices: {e}")
            return False

    def _batch_compress_layers(self, task: BatchCompressionTask):
        """
        Step 3: Compress all nodes for all layers using GPU kernels.

        This is shared across all compression stages and reuses the existing
        NodeCompressor.compress_layer() method.

        Args:
            task: The batch compression task with continuous tensors
        """
        num_layers = self.model_config.num_hidden_layers
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()

        try:
            # Compress all nodes for each layer in a single call
            for layer_idx in range(num_layers):
                self.compressor.compress_layer(
                    layer_idx=layer_idx,
                    kv_pool=kv_pool,
                    value=task.original_indices,          # Entire continuous tensor
                    out_cache_loc=task.compressed_indices, # Entire continuous tensor
                    top_budget=self.compressor.top_budget,
                    residual_budget=self.compressor.processed_residual_budget,
                    residual_heap_size=self.compressor.residual_heap_size,
                )

            logger.debug(f"Successfully compressed {len(task.node_li)} nodes across {num_layers} layers")

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            task.compressed_indices = None

    @abstractmethod
    def _update_metadata(self, task: BatchCompressionTask) -> int:
        """
        Step 4: Update metadata after compression (stage-specific).

        Must be implemented by subclasses to define how to update tree nodes,
        request indices, and other metadata.

        Args:
            task: The batch compression task

        Returns:
            Number of tokens saved by compression
        """
        pass

    def _is_node_compressible(self, node: "TreeNode") -> bool:
        """
        Check if a node is safe to compress.

        Common check used across different stages.

        Args:
            node: The tree node to check

        Returns:
            True if node can be compressed, False otherwise
        """
        if node.value is None:
            return False

        # Check node size matches radix_block_size
        node_size = node.value.size(0) if isinstance(node.value, torch.Tensor) else len(node.value)
        return node_size == self.radix_block_size
