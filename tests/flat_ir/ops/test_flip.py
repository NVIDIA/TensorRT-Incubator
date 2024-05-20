import numpy as np
import pytest

import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import FlipOp


# tensor shape, front-end flip dims, expected back-end flip dims
@pytest.fixture(
    params=[
        ((5,), None, (0,)),
        ((2, 3, 4), None, (0, 1, 2)),
        ((2, 3), 1, (1,)),
        # StableHLO does not care about the dim order as long as it's unique
        ((2, 3, 4), (1, 0), (1, 0)),
        # negative dims are converted at the front end
        ((2, 3, 4), -2, (1,)),
        ((2, 3, 4), (0, -1), (0, 2)),
    ],
    ids=[
        "vector_no_dims",
        "tensor_no_dims",
        "single_positive",
        "explicit_list",
        "single_negative",
        "list_with_negative",
    ],
)
def flip_params(request):
    return request.param


@pytest.fixture
def flat_ir(flip_params):
    shape, dims, _ = flip_params
    np_a = np.random.rand(*shape).astype(np.float32)
    a = tp.Tensor(np_a, shape=shape, dtype=tp.float32, name="a")
    out = tp.flip(a, dims=dims)
    out.name = "out"
    trace = Trace([out])
    return trace.to_flat_ir()


class TestFlipOp:
    def test_str(self, flat_ir, flip_params):
        shape, _, expected_dims = flip_params
        flip_op = flat_ir.ops[-1]
        assert isinstance(flip_op, FlipOp)
        target = (
            f"out: [shape=({', '.join(map(str, shape))},), dtype=(float32), loc=(gpu:0)]"
            + f" = FlipOp(a, dims=[{', '.join(map(str, expected_dims))}])"
        )
        assert str(flip_op) == target

    def test_mlir(self, flat_ir, flip_params):
        shape, _, expected_dims = flip_params
        # looking for a reverse op with the specified dims
        target = (
            rf"%1 = stablehlo.reverse %0, dims = [{', '.join(map(str, expected_dims))}]"
            + f" : tensor<{'x'.join(map(str, shape))}xf32>"
        )
        assert target in str(flat_ir.to_mlir())
