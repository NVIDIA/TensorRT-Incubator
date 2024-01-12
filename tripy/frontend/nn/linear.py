import numpy as np

from tripy.frontend.nn import Module
from tripy.frontend.nn import Parameter
from tripy.frontend.tensor import Tensor


class Linear(Module):
    """
    Applies linear transformation expressed by the equation y = x @ W^t + b.

    Args:
        in_features: dimensionality of input features
        out_features: dimensionality of output features
        bias: If set ``False``, then no bias will be used in linear tranformation. Default: ``True``.

    Returns:
        Output tensor with all dimensions except the last are of the same shape as the input and last dimension is equal to out_features.

    Example:
    ::

        import numpy as np

        a = tp.ones((2, 3))
        linear = tp.nn.Linear(3, 128)
        out = linear(a)

        assert out.numpy().shape == (2, 128)
    """

    def __init__(self, input_dims, output_dims, bias: bool = True):
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.tensor_ops import ones

        # Replace with random weights when #74 is completed.
        self.weight = Parameter(ones((output_dims, input_dims), dtype=float32))
        if bias:
            self.bias = Parameter(ones((1, output_dims), dtype=float32))

    def __call__(self, x):
        out = x @ (self.weight.transpose(1, 0))
        if self.__dict__["_params"]["bias"]:
            out = out + self.bias

        return out
