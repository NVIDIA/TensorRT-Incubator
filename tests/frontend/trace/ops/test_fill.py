from tripy.frontend.trace.ops import Fill
import tripy as tp


class TestFull:
    def test_op_func(self):
        a = tp.full([1, 2], 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)

    def test_shape_can_be_scalar(self):
        a = tp.full(2, 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)
        assert a.trace_tensor.producer.shape == (2,)


class TestFullLike:
    def test_op_func(self):
        t = tp.Tensor([[1, 2], [3, 4]])
        a = tp.full_like(t, 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)
