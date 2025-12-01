"""
Base Compression Engines (Legacy Module - Redirect)

This module has been refactored. All classes are now in engines.py.
This file exists for backward compatibility only.

For new code, import from:
    from sglang.srt.mem_cache.compression import NodeCompressor
    from sglang.srt.mem_cache.compression.engines import BaseCompressionEngine
"""

# Import all classes from the new module
from sglang.srt.mem_cache.compression.engines import (
    BaseCompressionEngine as BaseNodeCompressor,
    VanillaCompressionEngine,
    OptimizedCompressionEngine as OptimizedNodeCompressor,
    PreallocatingCompressionEngine as PreallocatingNodeCompressor,
)

__all__ = [
    "BaseNodeCompressor",
    "PreallocatingNodeCompressor",
    "OptimizedNodeCompressor",
]
