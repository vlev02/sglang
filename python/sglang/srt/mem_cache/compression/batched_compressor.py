"""
Batched Compression Engine (Legacy Module - Redirect)

This module has been refactored. BatchedNodeCompressor is now BatchedCompressionEngine in engines.py.
This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import NodeCompressor
    from sglang.srt.mem_cache.compression.engines import BatchedCompressionEngine
"""

from sglang.srt.mem_cache.compression.engines import BatchedCompressionEngine as BatchedNodeCompressor

__all__ = ["BatchedNodeCompressor"]
