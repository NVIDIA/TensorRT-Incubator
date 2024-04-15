import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import RandomNormal, RandomUniform


class TestRandom:
    def test_uniform(self):
        a = tp.rand((2, 3))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, RandomUniform)

    def test_uniform_non_float_dtype(self):
        with helper.raises(
            tp.TripyException,
            match="rand only supports float32 or float16.",
        ):
            a = tp.rand((2, 3), dtype=tp.int32)

    def test_normal(self):
        a = tp.randn((2, 3))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, RandomNormal)

    def test_normal_non_float_dtype(self):
        with helper.raises(
            tp.TripyException,
            match="randn only supports float32 or float16.",
        ):
            a = tp.randn((2, 3), dtype=tp.int32)
