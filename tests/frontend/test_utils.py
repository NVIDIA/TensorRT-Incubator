import tripy as tp
from tripy.frontend.utils import convert_inputs_to_tensors


@convert_inputs_to_tensors()
def func(a):
    return a


class TestConverInputsToTensors:
    def test_args(self):
        assert isinstance(func(0), tp.Tensor)

    def test_kwargs(self):
        assert isinstance(func(a=0), tp.Tensor)
