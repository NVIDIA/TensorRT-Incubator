---
name: tripy-debugging
description: 'Debug and diagnose errors in nvtripy code. Use when: interpreting TripyException stack traces, enabling MLIR/TensorRT debug output, understanding error reporting with stack_info, using raise_error, configuring debug environment variables, tracing compilation failures.'
---

# Debugging and Error Reporting in nvtripy

## When to Use

- Interpreting `TripyException` error messages and stack traces
- Enabling debug output for MLIR or TensorRT compilation
- Understanding the stack info system for precise error locations
- Adding error handling to new ops or modules
- Diagnosing compilation or runtime failures

## Error Reporting System

### `raise_error` — The Primary Error Function

From `nvtripy/common/exception.py`:

```python
from nvtripy.common.exception import raise_error

raise_error(
    "Brief description of the error.",
    details=[
        "Additional context line 1.",
        "Additional context line 2.",
        some_tensor,  # Will include the tensor's creation stack info
    ],
)
```

- First argument: The main error message (string)
- `details`: A list of strings and/or tensors. Tensors will have their stack info rendered.
- Raises `TripyException`

### Stack Info System

Every tensor captures its creation stack trace (`_stack_info`) for precise error reporting. This is managed in `nvtripy/utils/stack_info.py`.

When a tensor appears in `raise_error` details, the system renders the exact line and column where the tensor was created, helping users trace back to the problematic code.

Key points:
- `Tensor.from_trace_tensor(out, include_code_index=stack_depth)` — sets the stack depth for error reporting
- `STACK_DEPTH_OF_FROM_TRACE_TENSOR = 4` — the default depth in `create_op`
- `stack_depth_offset` parameter in `create_op` adjusts for wrapper functions

### Exception Hierarchy

```
TripyException (main user-facing exception)
└── Raised by raise_error() with formatted stack info
```

## Debug Configuration

### Environment Variables

Set these before running to enable debug output:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIPY_MLIR_DEBUG_ENABLED` | `"0"` | Enable MLIR debug output |
| `TRIPY_MLIR_DEBUG_TYPES` | `"-translate-to-tensorrt"` | Comma-separated MLIR pass types to debug |
| `TRIPY_MLIR_DEBUG_PATH` | `"/tripy/mlir-dumps"` | Directory for MLIR debug dumps |
| `TRIPY_TRT_DEBUG_ENABLED` | `"0"` | Enable TensorRT debug output |
| `TRIPY_TRT_DEBUG_PATH` | `"/tripy/tensorrt-dumps"` | Directory for TensorRT debug dumps |
| `TRIPY_EXTRA_ERROR_INFORMATION` | `""` | Comma-separated extra error info |

### Runtime Configuration

From `nvtripy/config.py`:

```python
import nvtripy as tp

# Timing cache (speeds up repeated compilations)
tp.config.timing_cache_file_path  # Default: /tmp/tripy-cache

# Input validation (disable for performance in production)
tp.config.enable_input_validation = False

# Extra error information
tp.config.extra_error_information = ["detailed"]
```

### Test Helper for Config Changes

```python
from tests import helper

with helper.config("enable_input_validation", False):
    # Code runs with validation disabled
    ...
# Automatically restored after the block
```

## Logging System

The logging system (`nvtripy/logging/`) provides granular control:

```python
import nvtripy as tp

# The global logger
logger = tp.logger

# Set verbosity for specific modules
logger.verbosity_trie.set("nvtripy.backend", "verbose")
```

The `VerbosityTrie` allows setting different log levels for different module paths, using a trie data structure for efficient prefix matching.

## Diagnosing Common Issues

### Constraint Validation Errors

Error pattern: `Expected 'input' to be one of [...] (but it was '...')`

This comes from the constraint system. Check:
1. The `input_requirements` in the `@wrappers.interface` decorator
2. The actual dtypes of the inputs being passed
3. Whether auto-casting should handle this case

### Compilation Errors

Enable MLIR debug output:
```bash
TRIPY_MLIR_DEBUG_ENABLED=1 python my_script.py
```

Check the dumps in `/tripy/mlir-dumps/` for the MLIR IR at each pass.

### Shape Mismatch Errors

The trace system tracks shapes through `infer_rank` on trace ops. Check:
1. The `infer_rank` policy on the trace op
2. Whether broadcasting is handled correctly
3. Dynamic dimensions (`DYNAMIC_DIM = -1`) vs static shapes

### Runtime Errors from Compiled Functions

Enable TensorRT debug output:
```bash
TRIPY_TRT_DEBUG_ENABLED=1 python my_script.py
```

Check:
- Input shapes fall within the `InputInfo` bounds
- Dynamic shapes are correctly configured with min/opt/max

## Adding Error Handling to New Code

### In Frontend Ops

```python
def my_op(input, dim):
    if input.rank < 2:
        raise_error(
            f"Input must have rank >= 2, but got rank: {input.rank}",
            details=[
                "Input is expected to have shape (N, *) where N is the batch size.",
                input,  # This renders the tensor's creation location
            ],
        )
```

### In Modules

```python
def forward(self, x):
    if self.quant_dtype is not None and self.weight_quant_dim == 1:
        raise_error(
            "Unsupported quantization parameters.",
            [
                "weight_quant_dim cannot be 1 when input_scale is provided.",
                f"input_scale={self.input_scale}, weight_quant_dim={self.weight_quant_dim}",
            ],
        )
```

## Testing Errors

Use the `helper.raises` context manager:

```python
from tests import helper
import nvtripy as tp

def test_invalid_dtype_fails():
    a = tp.Tensor([1.0, 2.0])
    b = tp.ones((2,), dtype=tp.float16)
    with helper.raises(tp.TripyException, match="Expected.*one of"):
        c = a + b
```

The `raises` helper supports:
- `ExcType`: Expected exception type
- `match`: Regex pattern to match against error message
- `has_stack_info_for`: Verify that specific tensors' stack info appears in the error

## Checklist

- [ ] Use `raise_error()` instead of raw `raise` for user-facing errors
- [ ] Include relevant tensors in `details` for stack info rendering
- [ ] Error message is actionable (says what went wrong AND what to do)
- [ ] Test error cases with `helper.raises(tp.TripyException, match=...)`
- [ ] Debug env vars documented if adding new debug output
