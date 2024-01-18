from tripy.frontend import Tensor
from tripy.frontend.ops.fill import Fill, full, full_like


class TestFull:
    def test_op_func(self):
        a = full([1, 2], 1)
        assert isinstance(a, Tensor)
        assert isinstance(a.op, Fill)

    def test_shape_can_be_scalar(self):
        a = full(2, 1)
        assert isinstance(a, Tensor)
        assert isinstance(a.op, Fill)
        assert a.op.shape == (2,)


class TestFullLike:
    def test_op_func(self):
        t = Tensor([[1, 2], [3, 4]])
        a = full_like(t, 1)
        assert isinstance(a, Tensor)
        assert isinstance(a.op, Fill)
