---
name: tripy-constraints
description: 'Author input/output constraints for nvtripy operations using the declarative constraint DSL. Use when: defining input_requirements or output_guarantees, writing @wrappers.interface decorators, auto-casting dtypes, using GetInput/GetReturn/OneOf/If/Equal, debugging constraint validation errors.'
---

# Authoring Constraints for nvtripy Operations

## When to Use

- Defining type constraints for a new or existing operation
- Writing `input_requirements` or `output_guarantees` for `@wrappers.interface`
- Debugging constraint validation errors at runtime
- Understanding auto-type-casting behavior

## Architecture Overview

The constraint system lives in `nvtripy/frontend/constraints/` and consists of:

- **Fetchers** (`fetcher.py`): Extract values from function arguments or return values
- **Logic** (`logic.py`): Compose constraints with boolean operators
- **Base** (`base.py`): Abstract base class for all constraints
- **Wrappers** (`nvtripy/frontend/wrappers.py`): The `@interface` decorator that applies constraints

## Core Components

### Fetchers — Extracting Values

```python
from nvtripy.frontend.constraints import GetInput, GetReturn

# Get a function parameter by name
GetInput("input")           # The parameter named "input"
GetInput("dtype")           # The parameter named "dtype"
GetInput("input").dtype     # The dtype of the "input" parameter (uses GetDataType)

# Get a return value by index
GetReturn(0)                # First return value
GetReturn(0).dtype          # Dtype of first return value
```

### Logic — Composing Constraints

```python
from nvtripy.frontend.constraints import OneOf, If, GetInput, GetReturn

# OneOf: value must be in a set
OneOf(GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16])

# Equal: two values must match
GetInput("weight").dtype == GetInput("input").dtype
GetReturn(0).dtype == GetInput("input").dtype

# NotEqual
GetInput("dtype") != None

# And: combine with &
OneOf(GetInput("input").dtype, [dt.float32, dt.float16])
& (GetInput("weight").dtype == GetInput("input").dtype)

# Or: combine with |
OneOf(GetInput("dtype"), [dt.float32]) | OneOf(GetInput("dtype"), [dt.float16])

# If: conditional constraint
If(
    GetInput("dtype") != None,                    # condition
    OneOf(GetInput("dtype"), [dt.float32]),        # then: applied when condition is true
    # else branch is optional
)

# Invert with ~
~OneOf(GetInput("dtype"), [dt.float32])  # dtype must NOT be float32
```

### All Available Logic Classes

| Class | Usage | Description |
|-------|-------|-------------|
| `OneOf(fetcher, options)` | `OneOf(GetInput("x").dtype, [dt.float32, dt.float16])` | Value must be in the list |
| `Equal` | `GetInput("a").dtype == GetInput("b").dtype` | Two values must be equal (created via `==`) |
| `NotEqual` | `GetInput("dtype") != None` | Two values must not be equal (created via `!=`) |
| `And` | `constraint1 & constraint2` | Both must be satisfied (created via `&`) |
| `Or` | `constraint1 \| constraint2` | At least one must be satisfied (created via `\|`) |
| `If(cond, then, else_)` | `If(GetInput("dtype") != None, then_constraint)` | Conditional constraint |
| `AlwaysTrue` | `AlwaysTrue()` | Always passes |
| `AlwaysFalse` | `AlwaysFalse()` | Always fails |

## Using `@wrappers.interface`

The `@wrappers.interface` decorator from `nvtripy/frontend/wrappers.py` accepts:

```python
@wrappers.interface(
    input_requirements=<Logic>,       # Pre-execution: validate inputs
    output_guarantees=<Logic>,        # Post-execution: validate outputs
    convert_to_tensors=True,          # Auto-convert TensorLike to Tensor
    conversion_preprocess_func=None,  # Custom preprocessing before conversion
)
```

- **`input_requirements`**: Checked BEFORE the function runs. If a dtype mismatch is found and auto-casting can fix it, the system will automatically cast inputs.
- **`output_guarantees`**: Checked AFTER the function runs. Verifies the output properties match expectations.

## Common Patterns

### Simple dtype restriction

```python
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16]),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def my_op(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
```

### Multiple inputs with matching dtypes

```python
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16])
    & (GetInput("weight").dtype == GetInput("input").dtype)
    & (GetInput("bias").dtype == GetInput("input").dtype),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def layernorm(input, weight, bias, eps):
```

### Optional dtype parameter

```python
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("input").dtype,
        [dt.float32, dt.float16, dt.bfloat16, dt.float8, dt.int8, dt.int32, dt.int64, dt.bool],
    )
    & If(
        GetInput("dtype") != None,
        OneOf(GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]),
    ),
    output_guarantees=If(
        GetInput("dtype") != None,
        GetReturn(0).dtype == GetInput("dtype"),
        GetReturn(0).dtype == GetInput("input").dtype,
    ),
)
def ones_like(input, dtype=None):
```

### Initializer ops (no tensor inputs, just dtype)

```python
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]
    ),
    output_guarantees=GetReturn(0).dtype == GetInput("dtype"),
)
def ones(shape, dtype=dt.float32):
```

## How Auto-Casting Works

When `input_requirements` include dtype constraints via `OneOf`:

1. The system checks if all inputs satisfy constraints
2. If a dtype mismatch is found, it looks for a valid target dtype from the `OneOf` options
3. Inputs are automatically cast to the matching dtype before the function executes

This means users don't need to manually cast, e.g., `tp.ones((2,), dtype=tp.float16) + tp.ones((2,), dtype=tp.float32)` will auto-cast.

## Constraint Error Messages

When constraints fail, the system generates an error like:
```
Expected 'input' to be one of [float32, float16, bfloat16] (but it was 'int32')
```

The error text comes from the `__str__` and `doc_str` methods of each `Logic` class.

## Checklist

- [ ] `input_requirements` covers all valid input dtypes with `OneOf`
- [ ] Multi-input ops require matching dtypes with `==` constraints
- [ ] Optional parameters guarded with `If(GetInput("x") != None, ...)`
- [ ] `output_guarantees` specify the output dtype relationship
- [ ] `&` used to combine multiple requirements (not nested `And()` calls)
- [ ] Test both valid and invalid dtype combinations
