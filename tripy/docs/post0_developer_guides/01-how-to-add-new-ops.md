<!-- Use the TEST: USE_PYTEST marker since we'll be defining unit tests as part of the guide.
    With this marker, those tests can actually be run under pytest.
    -->
<!-- Tripy: TEST: USE_PYTEST Start -->

# Adding New Operations

:::{seealso}
The [architecture guide](project:./00-architecture.md) provides an overview of the codebase.
:::

Adding a new operation involves:

- [Implementing a Trace operation](#implementing-the-trace-operation) **if** enabling a **new** tensorrt dialect op.

- [Implementing a frontend API](#implementing-the-frontend-api).


Let's implement Top-K, which we'll call Top-N so as not to collide with the actual Top-K operation:

## Implementing The Trace Operation

Trace operations implement the [`TraceOp`](source:/nvtripy/trace/ops/base.py) interface
and are located under [nvtripy/trace/ops](source:/nvtripy/trace/ops/).

:::{note}
[TensorRTOps.td](https://github.com/NVIDIA/TensorRT-Incubator/blob/main/mlir-tensorrt/tensorrt/include/mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.td)
defines the tensorrt dialect. Refer to that file for details on each operation.
:::

```py
# doc: no-eval
# nvtripy/trace/ops/topn.py
from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.common import datatype
from nvtripy.trace.ops.base import TraceOp

@dataclass(repr=False)
class TopN(TraceOp):
    # Attributes of the operation are added to the constructor by default.
    # Use `dataclasses.field(..., init=False)` to avoid that.
    n: int
    dim: int

    def infer_rank(self):
        # Top-K does not change the rank of its input
        rank = self.inputs[0].rank
        self.outputs[0].rank = rank
        self.outputs[1].rank = rank

    def infer_dtypes(self):
        # First output is top-n values, second is indices
        self.outputs[0].dtype = self.inputs[0].dtype
        self.outputs[1].dtype = datatype.int32

    # This is only required if `num_outputs != 1`:
    def get_num_outputs(self):
        return 2

    def to_mlir(self, inputs, outputs):
        # This method should *not* access `self.inputs` or `self.outputs`, only
        # `inputs` and `outputs`. The former are trace tensors while the latter
        # are MLIR operands.
        #
        # NOTE: If the MLIR API returned only a single tensor, we would need to
        # wrap it in a list.
        return tensorrt.top_k(inputs[0], self.n, self.dim, tensorrt.TopKOperationAttr.get("kMAX"))
```

We can add tests under [tests/trace/ops/](source:/tests/trace/ops/):

<!-- Tripy: DOC: OMIT Start -->
<!-- Need to simulate TopN being added to trace.ops module -->

```py
# doc: no-eval
import sys

class topn:
    TopN = TopN

sys.modules["nvtripy.trace.ops.topn"] = topn
```
<!-- Tripy: DOC: OMIT End -->

```py
# doc: no-eval
# tests/trace/ops/topn.py
import nvtripy as tp

from nvtripy.trace.ops.topn import TopN


class TestTopN:
    def test_infer_rank(self):
        inp = tp.ones((2, 2, 3))
        values, indices = TopN([inp.trace_tensor], dim=2, n=2).outputs
        assert values.rank == inp.rank
        assert indices.rank == inp.rank

    def test_infer_dtypes(self):
        inp = tp.ones((2, 2, 3))
        values, indices = TopN([inp.trace_tensor], dim=2, n=2).outputs
        assert values.dtype == inp.dtype
        assert indices.dtype == tp.int32
```

## Implementing The Frontend API

Frontend APIs are implemented in [nvtripy/frontend/ops](source:/nvtripy/frontend/ops).

They should:

1. Use `@export.public_api` to export themselves into the `nvtripy` module.
    This also:
    - Controls where the API is documented.
    - Enables type checking and function overloading.

2. Use `@wrappers.interface` to express data type constraints.

3. Include documentation, including at least one code example.

```py
# doc: no-eval
# nvtripy/frontend/ops/topn.py
from typing import Tuple

from nvtripy import export
from nvtripy.trace.ops.topn import TopN
from nvtripy.frontend import wrappers
from nvtripy.frontend.ops import utils as op_utils

@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: ["T1", "T2"]},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "int64"], "T2": ["int32"]},
)
def topn(input: "nvtripy.Tensor", n: int, dim: int) -> Tuple["nvtripy.Tensor", "nvtripy.Tensor"]:
    # See docs/README.md for more information on how to write docstrings
    """
    Returns the top-n values in the tensor and their
    indices along the specified dimension.

    Args:
        input: The input tensor.
        n: The number of values to take.
        dim: The dimension along which to find the top-n values.

    Returns:
        The top-n values and indices

    .. code-block:: python
        :linenos:

        inp = tp.iota((1, 5), dim=1)
        values, indices = tp.topn(inp, n=2, dim=1)

        assert tp.equal(values, tp.Tensor([[4.0, 3.0]]))
        assert tp.equal(indices, tp.Tensor([[4, 3]]))
    """
    # The `process_dim` helper performs bounds checking and handles
    # negative dimensions:
    dim = op_utils.process_dim(dim, input.rank)

    # The variadic arguments to `create_op` should match the attributes
    # of the trace operation.
    return op_utils.create_op(TopN, [input], n=n, dim=dim)
```

We can add tests in [tests/frontend/ops/](source:/tests/frontend/ops/) to test the frontend
function, e.g. parameter bounds checking:

```py
# doc: no-eval
# tests/frontend/ops/test_topN.py
import nvtripy as tp
from tests import helper

class TestTopN:
    def test_invalid_dim(self):
        inp = tp.ones((5, 5))
        with helper.raises(tp.TripyException, match="Dimension argument is out of bounds"):
            values, indices = tp.topn(inp, n=1, dim=3)
```


We can add integration tests in [tests/integration/](source:/tests/integration)
to test end-to-end functionality and accuracy:

```py
# doc: no-eval
# tests/integration/test_topN.py
import nvtripy as tp

# When implementing a real operation, we would likely want
# more exhaustive testing:
def test_topN():
    # TensorRT requires 2 dimensions for top-n:
    inp = tp.unsqueeze(tp.arange(5) + 2.0, dim=1)
    values, indices = tp.topn(inp, n=1, dim=0)

    # The last value in `arange` will be the largest:
    assert tp.equal(values, tp.Tensor([[6.0]]))
    assert tp.equal(indices, tp.Tensor([[4]]))
```

<!-- Tripy: TEST: USE_PYTEST End -->
