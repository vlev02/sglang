"""
Insertion Batch Processor (Legacy Module - Redirect)

This module has been refactored. InsertionCacheCompressor is now InsertionBatchProcessor in batch_processors.py.
This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import InsertionCacheCompressor
"""

from sglang.srt.mem_cache.compression.tasks import InsertionCompressionTask
from sglang.srt.mem_cache.compression.batch_processors import InsertionBatchProcessor as InsertionCacheCompressor

__all__ = [
    "InsertionCacheCompressor",
    "InsertionCompressionTask",
]
