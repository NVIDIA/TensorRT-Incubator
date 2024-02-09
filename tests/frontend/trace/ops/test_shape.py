import tripy as tp


def test_shape():
    a = tp.ones((3, 4))
    shape_a = a.shape
    assert isinstance(a, tp.Tensor)
    assert isinstance(shape_a, tp.Tensor)
