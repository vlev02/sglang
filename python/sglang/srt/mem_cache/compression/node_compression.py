"""
Node Compression Public API (Legacy Module - Redirect)

This module has been refactored. All compression engines are now in engines.py.
This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import NodeCompressor
"""

from sglang.srt.mem_cache.compression.engines import (
    BaseCompressionEngine as BaseNodeCompressor,
    VanillaCompressionEngine as VanillaNodeCompressor,
    OptimizedCompressionEngine as OptimizedNodeCompressor,
    PreallocatingCompressionEngine as PreallocatingNodeCompressor,
    BatchedCompressionEngine as UltraOptimizedNodeCompressor,
)

# Default compressor
NodeCompressor = UltraOptimizedNodeCompressor

__all__ = [
    "BaseNodeCompressor",
    "VanillaNodeCompressor",
    "OptimizedNodeCompressor",
    "PreallocatingNodeCompressor",
    "UltraOptimizedNodeCompressor",
    "NodeCompressor",
]
