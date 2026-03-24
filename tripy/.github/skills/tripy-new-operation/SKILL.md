---
name: tripy-new-operation
description: 'Add a new operation to nvtripy. Use when: implementing a new op, adding a frontend op, creating a trace op, registering an op in the API. Covers the full Frontend → Trace → MLIR pipeline including export decorators, constraint definitions, and init registration.'
---

# Adding a New Operation to nvtripy

## When to Use

- Adding a new mathematical, tensor, or neural network operation
- Creating a new frontend function that maps to TensorRT/MLIR ops
- Extending the op registry with unary, binary, or custom operations

## Architecture Overview

Operations in nvtripy follow a **Frontend → Trace → MLIR** pipeline:

1. **Trace Op** (`nvtripy/trace/ops/`): Defines the computational graph node — rank inference, dtype inference, and MLIR code generation.
2. **Frontend Op** (`nvtripy/frontend/ops/`): The public API function — exports, constraints, docstring, and bridges to the trace op via `create_op()`.
3. **Registration**: Both `__init__.py` files must be updated so the op is discoverable.

## Procedure

### Step 1: Create the Trace Operation

Create a file in `nvtripy/trace/ops/<op_name>.py`:

```python
from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class MyOp(TraceOp):
    # Add any op-specific parameters as dataclass fields:
    dim: int

    # Choose a rank inference policy:
    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def to_mlir(self, inputs, outputs):
        # Generate MLIR using the tensorrt dialect:
        return [tensorrt.some_op(inputs[0], self.dim)]
```

**Key base class requirements** (from `TraceOp`):

- `infer_rank` (required): Set output rank. Use policies from `InferRankPolicies`:
  - `same_as_input(idx=0)` — output rank matches input[idx]
  - `same_shape_as_input(idx=0)` — output has same shape (not just rank)
  - `same_as_shape_of_shape_input(idx=0)` — rank from a shape tensor
  - `max_of_inputs()` — rank is max across all inputs
  - Or define a custom function
- `to_mlir(self, inputs, outputs)` (required): Return list of MLIR operations
- `infer_dtypes()` (optional): Default propagates from `inputs[0]`. Override for multi-dtype ops.
- `infer_devices()` (optional): Default sets all outputs to GPU.
- `get_num_outputs()` (optional): Default is 1. Override for multi-output ops.
- `str_skip_fields()` (optional): Fields to omit from string representation.

**Factory pattern** for families of similar ops (see `trace/ops/unary.py`):

```python
def make_unary_op(name, attr_name):
    @dataclass(repr=False)
    class UnaryOp(TraceOp):
        infer_rank = op_utils.InferRankPolicies.same_as_input()

        def to_mlir(self, inputs, outputs):
            return [tensorrt.unary(inputs[0], tensorrt.UnaryOperationAttr.get(attr_name))]

    UnaryOp.__name__ = name
    return UnaryOp

Exp = make_unary_op("Exp", "kEXP")
```

### Step 2: Create the Frontend Operation

Create a file in `nvtripy/frontend/ops/<op_name>.py`:

```python
from typing import Optional

from nvtripy import export
from nvtripy.common import datatype as dt
from nvtripy.frontend import wrappers
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.my_op import MyOp


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16]),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def my_op(input: "nvtripy.Tensor", dim: Optional[int] = None) -> "nvtripy.Tensor":
    r"""
    Brief description of what the op does.

    Args:
        input: The input tensor.
        dim: The dimension to operate on.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.iota([2, 3], dtype=tp.float32)
        output = tp.my_op(input, dim=0)

        assert tp.allclose(output, expected_tensor)
    """
    dim = op_utils.process_dim(dim, input.rank)
    return op_utils.create_op(MyOp, [input], dim=dim)
```

**Key decorator details:**

- `@export.public_api(document_under="...")`: Registers in public API and docs hierarchy. Common paths:
  - `"operations/functions"` — general tensor ops
  - `"operations/initializers"` — tensor creation ops (ones, zeros, full)
  - `"operations/modules"` — nn module classes
- `@wrappers.interface(...)`: Defines input constraints and output guarantees (see constraint skill)
- Bridge to trace via `op_utils.create_op(TraceOpClass, [inputs], **kwargs)`

### Step 3: Register in `__init__.py` Files

**`nvtripy/frontend/ops/__init__.py`**: Add import so auto-discovery finds the module.

**`nvtripy/trace/ops/__init__.py`**: Usually empty — trace ops are imported directly by frontend ops.

### Step 4: Add as Tensor Method (Optional)

If the op should be callable as `tensor.my_op()`, register it in the `TENSOR_METHOD_REGISTRY` via the frontend tensor metaclass system. Check `nvtripy/frontend/tensor.py` for the pattern.

## Complete Example: Softmax

**Trace op** (`nvtripy/trace/ops/softmax.py`):

```python
@dataclass(repr=False)
class Softmax(TraceOp):
    dim: int
    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def to_mlir(self, inputs, outputs):
        return [tensorrt.softmax(inputs[0], self.dim)]
```

**Frontend op** (`nvtripy/frontend/ops/softmax.py`):

```python
@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16]),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def softmax(input: "nvtripy.Tensor", dim: Optional[int] = None) -> "nvtripy.Tensor":
    # Handle None dim by flattening
    # Handle rank < 2 by unsqueezing (TensorRT requirement)
    dim = op_utils.process_dim(dim, input.rank)
    return op_utils.create_op(Softmax, [input], dim=dim)
```

## Checklist

- [ ] Trace op created in `nvtripy/trace/ops/` with `infer_rank` and `to_mlir`
- [ ] Frontend op created in `nvtripy/frontend/ops/` with `@export.public_api` and `@wrappers.interface`
- [ ] Constraints defined for valid dtypes and output guarantees
- [ ] Docstring includes Args, Returns, and a working `.. code-block:: python` example
- [ ] `__init__.py` updated if needed for auto-discovery
- [ ] Tests added in `tests/frontend/ops/` and `tests/trace/ops/` (see testing skill)
