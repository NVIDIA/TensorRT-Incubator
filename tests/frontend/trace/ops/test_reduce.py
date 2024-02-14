import pytest

import tripy as tp
from tripy.frontend.trace.ops import Reduce, BinaryElementwise, ArgMinMax


class TestReduce:
    def test_sum(self):
        a = tp.ones((2, 3))
        a = a.sum(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Reduce)

    def test_max(self):
        a = tp.ones((2, 3))
        a = a.max(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Reduce)

    def test_mean(self):
        a = tp.ones((2, 3))
        a = a.mean(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, BinaryElementwise)
        assert a.op.kind == BinaryElementwise.Kind.DIV

    def test_variance(self):
        a = tp.ones((2, 3))
        a = a.var(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, BinaryElementwise)
        assert a.op.kind == BinaryElementwise.Kind.DIV

    def test_argmax(self):
        a = tp.ones((2, 3))
        a = a.argmax(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, ArgMinMax)

    def test_argmin(self):
        a = tp.ones((2, 3))
        a = a.argmin(0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, ArgMinMax)
