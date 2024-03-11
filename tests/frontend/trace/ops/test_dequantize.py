import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Dequantize


@pytest.fixture
def init_quant_tensor():
    a = tp.Tensor([1.0, 2.0])
    quant_a = tp.quantize(a, tp.int8, 1.0)
    return quant_a


class TestDequantize:
    def test_op(self, init_quant_tensor):
        a = init_quant_tensor
        a = tp.dequantize(a, tp.float32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Dequantize)

    def test_unsupported_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="input does not have a valid dtype to dequantize",
        ):
            a = tp.dequantize(a, tp.float32)
