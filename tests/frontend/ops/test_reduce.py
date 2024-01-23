import pytest

import tripy as tp
from tripy.frontend.ops.reduce import Reduce


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

    def test_invalid_argument(self):
        a = a = tp.ones((2, 3))

        with pytest.raises(tp.TripyException, match="Invalid combination of arguments.") as exc:
            a = a.max(keepdim=True)
        print(str(exc.value))
