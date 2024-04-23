import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Quantize


class TestQuantize:
    def test_op(self):
        a = tp.Tensor([1.0, 2.0])
        a = tp.quantize(a, 0.9, tp.int8)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Quantize)

    def test_invalid_input_dtype(self):
        a = tp.Tensor([1, 2], dtype=tp.int32)
        with helper.raises(
            tp.TripyException,
            match="Input does not have a valid dtype to quantize.",
        ):
            a = tp.quantize(a, 0.9, tp.int8)

    def test_unsupported_quant_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="Unsupported quantization dtype.",
        ):
            a = tp.quantize(a, 0.9, tp.float16)

    def test_infer_rank(self):
        a = tp.ones((2, 3))
        a = tp.quantize(a, 0.9, tp.int8)
        assert a.trace_tensor.rank == 2
