import pytest

import tripy as tp
from tripy.frontend.ops.iota import Iota


class TestIota:
    def test_iota(self):
        a = tp.iota([2, 3])
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Iota)

    def test_iota_like(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Iota)

    def test_invalid_dim(self):
        with pytest.raises(tp.TripyException, match="Invalid iota dim.") as exc:
            a = tp.iota([2, 3], dim=3)
        print(str(exc.value))
