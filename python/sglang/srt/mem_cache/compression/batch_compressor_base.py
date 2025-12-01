"""
Batch Compression Base Classes (Legacy Module - Redirect)

This module has been refactored:
- Tasks are now in tasks.py
- Processors are now in batch_processors.py

This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import (
        CompressionTask,
        InsertionCacheCompressor,
        NodeCacheCompressor,
    )
"""

# Import from new modules
from sglang.srt.mem_cache.compression.tasks import CompressionTask as BatchCompressionTask
from sglang.srt.mem_cache.compression.batch_processors import BaseBatchProcessor as BaseBatchCompressor

__all__ = [
    "BatchCompressionTask",
    "BaseBatchCompressor",
]
