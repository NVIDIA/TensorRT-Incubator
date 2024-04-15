import pytest

import tripy as tp
from tripy.frontend.trace.ops import Reduce, BinaryElementwise, ArgMinMax


class TestReduce:
    def test_sum(self):
        a = tp.ones((2, 3))
        a = tp.sum(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_max(self):
        a = tp.ones((2, 3))
        a = tp.max(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_mean(self):
        a = tp.ones((2, 3))
        a = tp.mean(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, BinaryElementwise)
        assert a.trace_tensor.producer.kind == BinaryElementwise.Kind.DIV

    def test_variance(self):
        a = tp.ones((2, 3))
        a = tp.var(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, BinaryElementwise)
        assert a.trace_tensor.producer.kind == BinaryElementwise.Kind.DIV

    def test_argmax(self):
        a = tp.ones((2, 3))
        a = tp.argmax(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, ArgMinMax)

    def test_argmin(self):
        a = tp.ones((2, 3))
        a = tp.argmin(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, ArgMinMax)
