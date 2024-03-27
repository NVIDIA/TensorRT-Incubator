import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Dequantize


class TestDequantize:

    def test_op(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        a = tp.dequantize(a, 0.9, tp.float32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Dequantize)

    def test_invalid_input_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="Input does not have a valid dtype to dequantize",
        ):
            a = tp.dequantize(a, 0.9, tp.float32)

    def test_invalid_dequant_dtype(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        with helper.raises(
            tp.TripyException,
            match="Invalid dequantization dtype.",
        ):
            a = tp.dequantize(a, 0.9, tp.int32)
