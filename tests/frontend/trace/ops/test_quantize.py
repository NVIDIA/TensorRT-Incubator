import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Quantize


class TestQuantize:
    def test_op(self):
        a = tp.Tensor([1, 2])
        a = tp.quantize(a, tp.int8, 1.0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Quantize)

    def test_unsupported_dtype(self):
        a = tp.Tensor([1, 2])
        with helper.raises(
            tp.TripyException,
            match="Unsupported quantization dtype.",
        ):
            a = tp.quantize(a, tp.float16, 1.0)
