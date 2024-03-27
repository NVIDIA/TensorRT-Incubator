import pytest

import tripy as tp
from tripy.frontend.trace.ops import Copy


@pytest.mark.parametrize("src, dst", [("cpu", "gpu"), ("gpu", "cpu")])
def test_copy(src, dst):
    a = tp.Tensor([1, 2], device=tp.device(src))
    a = tp.copy(a, tp.device(dst))
    assert isinstance(a, tp.Tensor)
    assert isinstance(a.op, Copy)
    assert a.op.target.kind == dst
