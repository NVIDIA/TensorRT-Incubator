import tripy as tp


class TestShape:
    def test_shape(self):
        a = tp.ones((3, 4))
        shape_a = a.shape
        assert isinstance(a, tp.Tensor)
        assert isinstance(shape_a, tp.Shape)

    def test_infer_rank(self):
        a = tp.ones((3, 4))
        shape_a = a.shape
        assert shape_a.trace_tensor.rank == 1
