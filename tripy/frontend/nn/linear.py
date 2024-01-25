from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


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

    .. code:: python
        :number-lines:

        a = tp.ones((2, 3))
        linear = tp.nn.Linear(3, 8)
        out = linear(a)

        print(out)
        assert out.numpy().shape == (2, 8)
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
        if hasattr(self, "bias"):
            out = out + self.bias

        return out
