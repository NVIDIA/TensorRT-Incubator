---
name: tripy-new-module
description: 'Add a new neural network module to nvtripy. Use when: creating an nn layer, implementing a Module subclass, adding a new layer like Linear/LayerNorm/Conv, defining parameters with DefaultParameter or OptionalParameter, using constant_fields decorator.'
---

# Adding a New Module to nvtripy

## When to Use

- Creating a new neural network layer (e.g., normalization, attention, convolution)
- Implementing a `Module` subclass with learnable parameters
- Adding a module that wraps existing ops into a reusable component

## Architecture Overview

Modules live in `nvtripy/frontend/module/` and follow this pattern:

1. **Optional helper function**: A standalone function (not exported) that implements the math, decorated with `@wrappers.interface` for constraints.
2. **Module class**: A `@dataclass` subclass of `Module` with `@export.public_api` and `@constant_fields`.
3. **Parameters**: Use `DefaultParameter` (must be set before use) or `OptionalParameter` (can be None).

## Procedure

### Step 1: Create the Module File

Create `nvtripy/frontend/module/<module_name>.py`:

```python
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from nvtripy import export, utils
from nvtripy.common import datatype
from nvtripy.frontend import wrappers
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter, OptionalParameter
from nvtripy.frontend.tensor import Tensor
from nvtripy.frontend.wrappers import constant_fields
from nvtripy.frontend.ops import utils as op_utils

# If needed, import the trace op:
from nvtripy.trace.ops.my_op import MyOp

from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf


# Optional: standalone function with constraints (used by the module's forward())
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [datatype.float32, datatype.float16, datatype.bfloat16])
    & (GetInput("weight").dtype == GetInput("input").dtype),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def my_layer_func(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: "nvtripy.Tensor",
    eps: float,
) -> "nvtripy.Tensor":
    # Implementation using existing ops or create_op
    return op_utils.create_op(MyOp, [input, weight, bias], eps=eps)


@export.public_api(document_under="operations/modules")
@dataclass
@constant_fields(["dtype"])
class MyLayer(Module):
    r"""
    Brief math description of the layer.

    :math:`\text{MyLayer}(x) = f(x, W, b)`
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Tensor
    r"""The weight parameter of shape :math:`[\text{features}]`."""

    bias: Optional[Tensor]
    r"""The bias parameter of shape :math:`[\text{features}]`."""

    eps: float
    """A small value for numerical stability."""

    def __init__(
        self,
        features: int,
        bias: bool = True,
        dtype: datatype.dtype = datatype.float32,
        eps: float = 1e-5,
    ) -> None:
        r"""
        Args:
            features: Size of the feature dimension.
            bias: Whether to include a bias term.
            dtype: The data type for parameters.
            eps: Small constant for numerical stability.

        .. code-block:: python
            :linenos:

            layer = tp.MyLayer(3)

            layer.weight = tp.iota(layer.weight.shape)
            layer.bias = tp.iota(layer.bias.shape)

            input = tp.iota((2, 3), dim=1)
            output = layer(input)

            assert cp.from_dlpack(output).get().shape == (2, 3)
        """
        super().__init__()

        self.dtype = dtype
        self.weight = DefaultParameter((features,), dtype=dtype)

        self.bias = None
        if bias:
            self.bias = DefaultParameter((features,), dtype=dtype)

        self.eps = eps

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        return my_layer_func(x, self.weight, self.bias, self.eps)
```

### Step 2: Understand Parameter Types

**`DefaultParameter(shape, dtype)`**: Creates a placeholder that MUST be replaced with real data before the module runs. Used for required weights:

```python
self.weight = DefaultParameter((out_features, in_features), dtype=dtype)
```

**`OptionalParameter(shape, dtype)`**: Can be `None` — used for optional weights like quantization scales:

```python
self.input_scale = OptionalParameter(shape=[], dtype=dtype)
```

### Step 3: Use `@constant_fields`

The `@constant_fields(["field1", "field2"])` decorator marks fields as compile-time constants. These fields will be baked into the compiled graph and cannot change at runtime. Use for:

- `dtype` — data type configuration
- `normalized_shape` — shape parameters that define the layer structure
- `quant_dtype` — quantization configuration

### Step 4: Register the Module

The `@export.public_api(document_under="operations/modules")` decorator handles registration. The module will be accessible as `tp.MyLayer(...)`.

Ensure the module file is imported in `nvtripy/frontend/module/__init__.py`.

## Complete Example: LayerNorm

```python
# Helper function with constraints
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [datatype.float32, datatype.float16, datatype.bfloat16])
    & (GetInput("weight").dtype == GetInput("input").dtype)
    & (GetInput("bias").dtype == GetInput("input").dtype),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def layernorm(input, weight, bias, eps):
    normalized_shape = weight.shape
    D = len(normalized_shape)
    input_rank = input.rank

    if input_rank < 2:
        raise_error(f"Input must have rank >= 2, got {input.rank}")

    if input_rank > D:
        broadcast_shape = (1,) * (input_rank - D) + normalized_shape
        weight = reshape(weight, broadcast_shape)
        bias = reshape(bias, broadcast_shape)

    return op_utils.create_op(LayerNormOp, [input, weight, bias],
                               normalized_shape=normalized_shape, eps=eps)


@export.public_api(document_under="operations/modules")
@dataclass
@constant_fields(["dtype", "normalized_shape"])
class LayerNorm(Module):
    dtype: datatype.dtype
    normalized_shape: Sequence[int]
    weight: Tensor
    bias: Tensor
    eps: float

    def __init__(self, normalized_shape, dtype=datatype.float32, eps=1e-5):
        super().__init__()
        self.dtype = dtype
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.weight = DefaultParameter(normalized_shape, dtype=dtype)
        self.bias = DefaultParameter(normalized_shape, dtype=dtype)
        self.eps = eps

    def forward(self, x):
        return layernorm(x, self.weight, self.bias, self.eps)
```

## Complete Example: Linear

Key patterns from `Linear`:

- Weight shape: `(out_features, in_features)` — transposed in `forward()`
- Optional bias with sentinel: `self.bias = DefaultParameter(...) if bias else None`
- Quantization support with `OptionalParameter` for scales
- Uses `@constant_fields(["dtype", "quant_dtype"])` for compile-time config

## Module Base Class Features

The `Module` base class (`nvtripy/frontend/module/module.py`) provides:

- `state_dict()`: Recursively collects all `Tensor` parameters (supports nested modules, lists, dicts)
- `load_state_dict(state_dict, strict=True)`: Loads parameters with shape/dtype validation
- `__setattr__`: Validates parameter compatibility on assignment
- `__call__`: Calls `forward()` — modules are callable like functions

## Checklist

- [ ] Module file created in `nvtripy/frontend/module/`
- [ ] Inherits from `Module`, decorated with `@dataclass` and `@export.public_api`
- [ ] `@constant_fields` applied for compile-time configuration fields
- [ ] `__init__` calls `super().__init__()` and uses `DefaultParameter`/`OptionalParameter`
- [ ] `forward()` method implemented
- [ ] Docstrings with math notation and working code examples
- [ ] Helper function with `@wrappers.interface` constraints (if applicable)
- [ ] Registered in `nvtripy/frontend/module/__init__.py`
- [ ] Tests added in `tests/frontend/module/`
