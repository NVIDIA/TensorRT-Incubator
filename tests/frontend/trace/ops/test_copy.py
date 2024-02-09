import pytest

import tripy
from tripy.common import device
from tripy.frontend.trace.ops import Copy


@pytest.mark.parametrize("src, dst", [("cpu", "gpu"), ("gpu", "cpu")])
def test_copy(src, dst):
    a = tripy.Tensor([1, 2], device=device(src))
    a = a.to(device(dst))
    assert isinstance(a, tripy.Tensor)
    assert isinstance(a.op, Copy)
    assert a.op.target.kind == dst
