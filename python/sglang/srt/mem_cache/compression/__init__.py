"""
KV Cache Compression Module

This module provides a clean, modular architecture for KV cache compression in SGLang.

Module Organization:
    tasks.py              - Data structures for compression tasks
    engines.py            - Core single-node compression engines
    batch_processors.py   - Batch-level compression coordinators

Architecture:

    ┌─────────────────────────────────────────────────────────────┐
    │ PUBLIC API (this file)                                       │
    │ - NodeCompressor (default compression engine)               │
    │ - InsertionCacheCompressor (insertion batch processor)      │
    │ - NodeCacheCompressor (decode batch processor)              │
    └─────────────────────────────────────────────────────────────┘
                                 │
    ┌────────────────────────────┴─────────────────────────────────┐
    │                                                              │
    ┌──────────────▼────────────────┐  ┌───────────▼──────────────┐
    │ engines.py                    │  │ batch_processors.py       │
    │ - BaseCompressionEngine       │  │ - BaseBatchProcessor      │
    │ - VanillaCompressionEngine    │  │ - InsertionBatchProcessor │
    │ - OptimizedCompressionEngine  │  │ - DecodeBatchProcessor    │
    │ - PreallocatingCompress...    │  │                           │
    │ - BatchedCompressionEngine    │  │                           │
    └────────────┬──────────────────┘  └───────────┬───────────────┘
                 │                                  │
                 └──────────────┬───────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ tasks.py              │
                    │ - CompressionTask     │
                    │ - InsertionComp...    │
                    │ - DecodeComp...       │
                    └───────────────────────┘

Key Design Principles:
1. Separation of Concerns: Tasks, engines, and processors are separate
2. Easy to Replace: Switch compression engines by changing one line
3. Clear Naming: Class names reflect their purpose and layer
4. Minimal Dependencies: Clean import structure

Usage Examples:

    # For single-node compression (RadixCache):
    from sglang.srt.mem_cache.compression import NodeCompressor

    compressor = NodeCompressor(budget=16, page_size=128)
    value, out = compressor.compress_node(value, kv_pool, out_cache_loc)

    # For batch compression (Scheduler):
    from sglang.srt.mem_cache.compression import (
        InsertionCacheCompressor,
        NodeCacheCompressor,
    )

    # Insertion stage
    insertion_comp = InsertionCacheCompressor(tree_cache, model_config)
    insertion_comp.begin_batch()
    # ... mark nodes ...
    insertion_comp.compress_batch()

    # Decode stage
    decode_comp = NodeCacheCompressor(tree_cache, q_win, model_config, stages)
    decode_comp.compress_batch(schedule_batch)

To Replace Compression Algorithm:

    # In this file, change:
    NodeCompressor = BatchedCompressionEngine  # Current
    # To:
    NodeCompressor = YourNewEngine  # Custom

    # That's it! All code automatically uses the new engine.
"""

# Import task data structures
from sglang.srt.mem_cache.compression.tasks import (
    CompressionTask,
    InsertionCompressionTask,
    DecodeCompressionTask,
)

# Import compression engines
from sglang.srt.mem_cache.compression.engines import (
    BaseCompressionEngine,
    VanillaCompressionEngine,
    OptimizedCompressionEngine,
    PreallocatingCompressionEngine,
    BatchedCompressionEngine,
)

# Import batch processors
from sglang.srt.mem_cache.compression.batch_processors import (
    BaseBatchProcessor,
    InsertionBatchProcessor,
    DecodeBatchProcessor,
)

# ============================================================================
# Public API - These are the names used by external code
# ============================================================================

# Default single-node compression engine
# To change the compression algorithm globally, modify this assignment
NodeCompressor = BatchedCompressionEngine

# Batch-level processors (keep existing names for compatibility)
InsertionCacheCompressor = InsertionBatchProcessor
NodeCacheCompressor = DecodeBatchProcessor

# Legacy compatibility aliases
# These maintain backward compatibility with existing code
BaseNodeCompressor = BaseCompressionEngine
PreallocatingNodeCompressor = PreallocatingCompressionEngine
OptimizedNodeCompressor = OptimizedCompressionEngine
UltraOptimizedNodeCompressor = BatchedCompressionEngine
VanillaNodeCompressor = VanillaCompressionEngine

# Batch compression task aliases
BatchCompressionTask = CompressionTask
BaseBatchCompressor = BaseBatchProcessor

# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # ===== PRIMARY PUBLIC API =====
    # These are the main interfaces external code should use
    "NodeCompressor",  # Default single-node compression engine
    "InsertionCacheCompressor",  # Insertion batch processor
    "NodeCacheCompressor",  # Decode batch processor
    # ===== TASK DATA STRUCTURES =====
    "CompressionTask",
    "InsertionCompressionTask",
    "DecodeCompressionTask",
    # ===== COMPRESSION ENGINES =====
    # For advanced users who want to select specific engines
    "BaseCompressionEngine",
    "VanillaCompressionEngine",
    "OptimizedCompressionEngine",
    "PreallocatingCompressionEngine",
    "BatchedCompressionEngine",
    # ===== BATCH PROCESSORS =====
    # For advanced users who want to customize batch processing
    "BaseBatchProcessor",
    "InsertionBatchProcessor",
    "DecodeBatchProcessor",
    # ===== LEGACY COMPATIBILITY =====
    # Kept for backward compatibility with existing code
    "BaseNodeCompressor",
    "PreallocatingNodeCompressor",
    "OptimizedNodeCompressor",
    "UltraOptimizedNodeCompressor",
    "VanillaNodeCompressor",
    "BatchCompressionTask",
    "BaseBatchCompressor",
]
