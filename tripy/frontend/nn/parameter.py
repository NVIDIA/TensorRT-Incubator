from tripy.frontend.tensor import Tensor


# todo: add metaclass to allow isintance(cls, Parameter) to return true if object is Tensor.
class Parameter(Tensor):
    """Parameters are regular tripy tensors along with an extra attribute that helps the underlying
    compiler to optimize the network for performance.
    """

    def __new__(cls, data=None):
        if data is None:
            # Create an empty tensor
            assert False, "tripy does not support creating an empty tensor as of 12/12/2023"
        data._is_param = True
        return data

    def __repr__(self) -> str:
        return repr(self.val)

    def __str__(self) -> str:
        return str(self.val)
