import pytest

import tripy as tp
from tripy.frontend.trace.ops import Copy


@pytest.mark.parametrize("src, dst", [("cpu", "gpu"), ("gpu", "cpu")])
def test_copy(src, dst):
    a = tp.Tensor([1, 2], device=tp.device(src))
    a = tp.copy(a, tp.device(dst))
    assert isinstance(a, tp.Tensor)
    assert isinstance(a.trace_tensor.producer, Copy)
    assert a.trace_tensor.producer.target.kind == dst
