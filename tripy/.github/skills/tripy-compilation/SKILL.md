---
name: tripy-compilation
description: 'Work with the nvtripy compilation pipeline. Use when: using tp.compile, creating InputInfo or DimensionInputInfo, understanding the Trace → MLIR → TensorRT flow, configuring optimization levels, working with Executable objects, debugging compilation, using dynamic shapes or NamedDimension.'
---

# nvtripy Compilation Pipeline

## When to Use

- Compiling functions or modules with `tp.compile()`
- Defining runtime inputs with `InputInfo` and `DimensionInputInfo`
- Working with compiled `Executable` objects
- Configuring compilation options (optimization level, timing cache)
- Understanding the Trace → MLIR → TensorRT flow
- Using dynamic shapes with min/opt/max bounds
- Debugging compilation failures

## Compilation Flow

```
User Function → Trace (graph) → MLIR (IR) → TensorRT (engine) → Executable
```

1. **Trace**: The function is called with tracer tensors to record the computation graph
2. **MLIR**: The trace graph is lowered to MLIR using the `tensorrt` dialect
3. **TensorRT**: MLIR is compiled to a TensorRT engine
4. **Executable**: The engine is wrapped in a callable `Executable` object

## `tp.compile()` — The Main Entry Point

```python
compiled_fn = tp.compile(
    func,                    # Function or Module to compile
    optimization_level=3,    # 0-5, higher = better runtime, longer compile
    args=[...],              # Positional arguments
    kwargs={...},            # Keyword arguments
)
```

### Argument Types

| Argument Type | Behavior |
|---------------|----------|
| `InputInfo(shape, dtype)` | Becomes a runtime input to the executable |
| `DimensionInputInfo(value_bounds)` | Becomes a runtime scalar dimension input |
| `Tensor` | Baked in as a compile-time constant |
| Any other type | Baked in as a compile-time constant |

The compiled `Executable` only accepts parameters that were `InputInfo`/`DimensionInputInfo` in the original `compile()` call.

## `InputInfo` — Tensor Runtime Inputs

```python
# Static shape
inp = tp.InputInfo(shape=(2, 4), dtype=tp.float32)
# shape_bounds: min=(2,4), opt=(2,4), max=(2,4)

# Dynamic dimensions (min, opt, max)
inp = tp.InputInfo(shape=((1, 2, 3), 4), dtype=tp.float32)
# First dim: min=1, opt=2, max=3; second dim: fixed at 4
# shape_bounds: min=(1,4), opt=(2,4), max=(3,4)

# Named dimensions (must be equal at runtime)
window = tp.NamedDimension("window", 3, 5, 7)
inp = tp.InputInfo(shape=(1, window, window), dtype=tp.float32)
# Both dims named "window" must have the same value at runtime
```

### `DimensionInputInfo` — Scalar Dimension Inputs

For functions that take scalar shape values as parameters:

```python
dim_info = tp.DimensionInputInfo(value_bounds=(1, 2, 4))
# min=1, opt=2, max=4
```

Used when a function parameter controls a reshape or dynamic shape operation.

## `Executable` — Running Compiled Functions

```python
# The executable's signature matches the InputInfo parameters
compiled_fn = tp.compile(add, args=[
    tp.InputInfo((2, 4), dtype=tp.float32),  # "a"
    tp.InputInfo((2, 4), dtype=tp.float32),  # "b"
])

# Call with evaluated tensors
a = tp.ones((2, 4), dtype=tp.float32).eval()
b = tp.ones((2, 4), dtype=tp.float32).eval()
result = compiled_fn(a, b)
```

### Key `Executable` properties

- `input_infos`: Dict of parameter name → `InputInfo`
- `stream`: The CUDA stream used for execution
- `__signature__`: Compatible with `inspect.signature()` for introspection

### Important: `.eval()` for inputs

Runtime inputs to compiled functions should be evaluated tensors (not lazy). Use `.eval()` to force evaluation before passing to the executable.

## Compiling Modules

```python
class MyModel(tp.Module):
    def __init__(self):
        super().__init__()
        self.linear = tp.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
# Load real weights before compiling
model.linear.weight = tp.Tensor(weight_data)
model.linear.bias = tp.Tensor(bias_data)

compiled_model = tp.compile(
    model,
    args=[tp.InputInfo(shape=(2, 3), dtype=tp.float32)],
)
```

When compiling a `Module`:
- The module's `state_dict()` entries are named for readable traces
- Weights become compile-time constants (baked into the engine)
- Only `InputInfo` arguments become runtime inputs

## Dynamic Shapes

### Basic dynamic dimensions

```python
compiled_add = tp.compile(
    add,
    args=[
        tp.InputInfo(shape=((1, 2, 3), 2), dtype=tp.float32),
        tp.InputInfo(shape=((1, 2, 3), 2), dtype=tp.float32),
    ],
)

# Works for any first-dim size in [1, 3]:
small = compiled_add(tp.ones((1, 2)).eval(), tp.ones((1, 2)).eval())
big = compiled_add(tp.ones((3, 2)).eval(), tp.ones((3, 2)).eval())
```

### Named dimensions for constraints

```python
window_size = tp.NamedDimension("window_size", 3, 5, 7)
inp = tp.InputInfo((1, window_size, window_size), dtype=tp.float32)
# Both dimensions named "window_size" must be equal at runtime
```

### Scalar dimension inputs

```python
def dynamic_reshape(x, s):
    return tp.reshape(x, (-1, s))

compiled_reshape = tp.compile(
    dynamic_reshape,
    args=[
        tp.InputInfo(shape=(3, (2, 4, 6)), dtype=tp.float32),
        tp.DimensionInputInfo(value_bounds=(1, 2, 4)),
    ],
)

result = compiled_reshape(tp.ones((3, 4)).eval(), tp.DimensionSize(2))
assert result.shape == (6, 2)
```

## Compilation Options

### Optimization Level

| Level | Description |
|-------|-------------|
| 0 | Minimal optimization, fastest compile |
| 1–2 | Moderate optimization |
| 3 | Default — good balance |
| 4–5 | Maximum optimization, slowest compile |

### Timing Cache

```python
tp.config.timing_cache_file_path = "/path/to/cache"
```

The timing cache stores kernel profiling data across compilations, significantly speeding up repeated compilations with similar operations.

## Compiler Internals

The MLIR compiler (`nvtripy/backend/mlir/compiler.py`) uses these options:

- `--tensorrt-timing-cache-path`: Path to timing cache
- `--tensorrt-builder-opt-level`: Optimization level (0-5)
- `--force-entrypoints-return-allocs`: Memory management
- `--mlir-elide-elementsattrs-if-larger`: Debug readability
- `--tensorrt-layer-info-dir`: TensorRT layer debug info

## Function Requirements for `tp.compile`

The function passed to `compile()` must:

1. **Be pure** — no side effects (`print`, `assert`, file I/O)
2. **Return Tensor(s)** — only `Tensor` return types supported
3. **No collection inputs** — `List[Tensor]` or `Dict[str, Tensor]` will be frozen as constants
4. **No variadic args** — `*args` and `**kwargs` are frozen at compile time

## Checklist

- [ ] `InputInfo` used for all runtime tensor inputs
- [ ] `DimensionInputInfo` used for scalar shape parameters
- [ ] Dynamic dimension bounds specified as `(min, opt, max)` tuples
- [ ] Module weights loaded before calling `tp.compile()`
- [ ] Function is pure (no side effects)
- [ ] Runtime inputs `.eval()`'d before passing to executable
- [ ] Timing cache configured for repeated compilations
- [ ] Optimization level appropriate for use case
