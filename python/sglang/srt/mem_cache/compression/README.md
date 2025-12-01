# KV Cache Compression Module

## Overview

This module provides a clean, modular architecture for KV cache compression in SGLang. The refactored design separates concerns into distinct modules, making it easy to understand, maintain, and extend.

## Architecture

```
compression/
├── README.md                  # This file
├── __init__.py                # Public API and exports
├── tasks.py                   # Data structures for compression tasks
├── engines.py                 # Core single-node compression engines
├── batch_processors.py        # Batch-level compression coordinators
├── gpu_compression_optimizations.py  # GPU-accelerated variants (optional)
│
└── Legacy Compatibility Files (redirect to new modules):
    ├── base_compressor.py
    ├── batch_compressor_base.py
    ├── batched_compressor.py
    ├── insertion_compressor.py
    ├── node_cache_compressor.py
    └── node_compression.py
```

## Module Organization

### 1. `tasks.py` - Data Structures

Contains dataclass definitions for compression tasks:

- **`CompressionTask`**: Base task with node list and tensor indices
- **`InsertionCompressionTask`**: For insertion/prefill stages
- **`DecodeCompressionTask`**: For decode stages with request metadata

**Design Principle**: Keep data separate from logic.

### 2. `engines.py` - Core Compression Engines

Contains single-node compression implementations:

- **`BaseCompressionEngine`** (ABC): Abstract base class
- **`VanillaCompressionEngine`**: Original PyTorch implementation
- **`OptimizedCompressionEngine`**: Memory-optimized with cleanup
- **`PreallocatingCompressionEngine`**: Pre-allocated GPU buffers
- **`BatchedCompressionEngine`**: Vectorized batch processing (default)

**Design Principle**: Single responsibility - compress one node at a time.

### 3. `batch_processors.py` - Batch Coordinators

Contains batch-level compression orchestrators:

- **`BaseBatchProcessor`** (ABC): Abstract 4-step pipeline
- **`InsertionBatchProcessor`**: For insertion/prefill stages
- **`DecodeBatchProcessor`**: For decode stages

**Design Principle**: Coordinate multi-node compression across pipeline stages.

### 4. `__init__.py` - Public API

Exports the primary interfaces:

```python
# Default compression engine
NodeCompressor = BatchedCompressionEngine

# Batch processors
InsertionCacheCompressor = InsertionBatchProcessor
NodeCacheCompressor = DecodeBatchProcessor
```

**Design Principle**: One-line configuration to change compression algorithm.

## Usage Examples

### Single-Node Compression (RadixCache)

```python
from sglang.srt.mem_cache.compression import NodeCompressor

# Create compressor
compressor = NodeCompressor(
    budget=16,              # Compress to 16 tokens
    page_size=128,          # Original page size
    head_num=32,            # Number of attention heads
    head_dim=128,           # Dimension per head
    dtype=torch.float16,
    device=torch.device('cuda'),
)

# Compress a node
original_value, compressed_out = compressor.compress_node(
    value=node_indices,      # [128] original KV indices
    kv_pool=kv_cache_pool,   # KV cache pool
    out_cache_loc=out_locs,  # [16] pre-allocated output
)
```

### Batch Compression (Insertion Stage)

```python
from sglang.srt.mem_cache.compression import InsertionCacheCompressor

# Create processor
processor = InsertionCacheCompressor(tree_cache, model_config)

# Begin batch
processor.begin_batch()

# Mark nodes for compression during insertion
for node in newly_inserted_nodes:
    processor.mark_node_for_compression(node)

# Compress all marked nodes in one batch
num_compressed = processor.compress_batch()
```

### Batch Compression (Decode Stage)

```python
from sglang.srt.mem_cache.compression import NodeCacheCompressor

# Create processor
processor = NodeCacheCompressor(
    tree_cache=tree_cache,
    q_win_size=128,
    model_config=model_config,
    compress_stages="0110",  # After prefill and during decode
)

# Compress batch
num_compressed = processor.compress_batch(schedule_batch)
```

## How to Replace the Compression Algorithm

To use a different compression engine globally:

1. **Edit `__init__.py`:**

```python
# Change this line:
NodeCompressor = BatchedCompressionEngine  # Current

# To:
NodeCompressor = YourCustomEngine  # Your new engine
```

2. **That's it!** All code automatically uses the new engine.

3. **Or create a custom engine:**

```python
# In engines.py
class YourCustomEngine(BaseCompressionEngine):
    def compress_layer(self, layer_idx, kv_pool, value, out_cache_loc, ...):
        # Your custom compression logic
        pass
```

## Key Design Principles

### 1. Separation of Concerns

- **Tasks** (data) are separate from **Engines** (compression logic)
- **Engines** (single-node) are separate from **Processors** (batch coordination)
- Each module has a single, well-defined responsibility

### 2. Easy to Replace

- Changing the compression algorithm requires modifying ONE line in `__init__.py`
- All compression engines implement the same interface (`BaseCompressionEngine`)
- Drop-in replacement without touching external code

### 3. Clear Naming

- **Engine** suffix: Single-node compression (e.g., `BatchedCompressionEngine`)
- **Processor** suffix: Batch coordination (e.g., `DecodeBatchProcessor`)
- **Task** suffix: Data structures (e.g., `CompressionTask`)

### 4. Minimal Dependencies

- Engines only depend on PyTorch and KV pool
- Processors depend on engines and tasks
- Public API ties everything together
- Clean, acyclic dependency graph

### 5. Backward Compatibility

- Legacy files redirect to new modules
- Existing imports continue to work
- Gradual migration path for external code

## Compression Pipeline (4 Steps)

All batch processors follow the same 4-step pipeline:

### Step 1: Collect Tasks (Stage-Specific)

```python
def _collect_tasks(self, ...) -> Optional[CompressionTask]:
    # Identify nodes to compress
    # Build continuous tensors
    # Return task or None
```

### Step 2: Allocate Indices (Shared)

```python
def _allocate_indices(self, task: CompressionTask) -> bool:
    # Allocate continuous tensor for compressed indices
    # Update task.compressed_indices
```

### Step 3: Compress Layers (Shared)

```python
def _compress_layers(self, task: CompressionTask):
    # Call engine.compress_layer() for each layer
    # GPU operation using continuous tensors
```

### Step 4: Update Metadata (Stage-Specific)

```python
def _update_metadata(self, task: CompressionTask) -> int:
    # Update node values
    # Update tree cache stats
    # Free original KV cache
```

## Performance Notes

- **`BatchedCompressionEngine`** (default): Vectorized, eliminating Python loops
- **`PreallocatingCompressionEngine`**: Pre-allocated buffers, reduced allocation overhead
- **`OptimizedCompressionEngine`**: Explicit cleanup, better memory management
- **`VanillaCompressionEngine`**: Original implementation, for reference

Typical speedup: **2-3x** with `BatchedCompressionEngine` vs `VanillaCompressionEngine`

## Testing

```bash
# Test new module imports
python -c "from sglang.srt.mem_cache.compression import NodeCompressor; print('✅')"

# Test backward compatibility
python -c "from sglang.srt.mem_cache.compression.base_compressor import BaseNodeCompressor; print('✅')"

# Run compression tests
python tests/test_batched_compression.py
python tests/test_compression_head_scaling.py
```

## Migration Guide

### For New Code

Use the new public API:

```python
from sglang.srt.mem_cache.compression import (
    NodeCompressor,
    InsertionCacheCompressor,
    NodeCacheCompressor,
)
```

### For Existing Code

Legacy imports still work:

```python
# Old (still works)
from sglang.srt.mem_cache.compression.base_compressor import BaseNodeCompressor
from sglang.srt.mem_cache.compression.node_compression import NodeCompressor

# New (recommended)
from sglang.srt.mem_cache.compression import BaseCompressionEngine, NodeCompressor
```

### Deprecation Timeline

- **Current**: All legacy imports work via redirects
- **Future**: Legacy files may be marked as deprecated
- **Long-term**: Legacy files may be removed (after migration period)

## Contributing

When adding a new compression algorithm:

1. Create a new engine class in `engines.py`:

```python
class MyNewEngine(BaseCompressionEngine):
    def compress_layer(self, ...):
        # Implementation
        pass
```

2. Export it in `__init__.py`:

```python
__all__ = [
    # ... existing exports ...
    "MyNewEngine",
]
```

3. (Optional) Make it the default:

```python
NodeCompressor = MyNewEngine
```

4. Add tests to `tests/test_*.py`

## License

Same as SGLang (Apache 2.0)
