"""
Decode Batch Processor (Legacy Module - Redirect)

This module has been refactored. NodeCacheCompressor is now DecodeBatchProcessor in batch_processors.py.
This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import NodeCacheCompressor
"""

from sglang.srt.mem_cache.compression.tasks import DecodeCompressionTask
from sglang.srt.mem_cache.compression.batch_processors import DecodeBatchProcessor as NodeCacheCompressor

# Legacy alias
CompressionTask = DecodeCompressionTask

__all__ = [
    "NodeCacheCompressor",
    "DecodeCompressionTask",
    "CompressionTask",  # Legacy alias
]
