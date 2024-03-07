from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


class Linear(Module):
    r"""
    Applies a linear transformation to the input:

    :math:`Linear(x) = xW^T + b`

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        bias: Whether to include the bias term.

    .. code-block:: python
        :linenos:
        :caption: Example

        linear = tp.nn.Linear(3, 4)

        input = tp.iota((2, 3))
        output = linear(input)

        assert output.numpy().shape == (2, 4)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.ops import ones

        # Replace with random weights when #74 is completed.
        self.weight: Parameter = Parameter(ones((out_features, in_features), dtype=float32))
        r"""The :math:`W` matrix of shape :math:`[\text{out_features}, \text{in_features}]`"""

        if bias:
            self.bias: Parameter = Parameter(ones((out_features), dtype=float32))
            r"""The :math:`b` matrix of shape :math:`[1, \text{out_features}]`"""

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor, of shape :math:`[*, \text{in_features}]`.

        Returns:
            A tensor of shape :math:`[*, \text{out_features}]`.
        """
        out = x @ (self.weight.transpose(1, 0))
        if hasattr(self, "bias"):
            out = out + self.bias.unsqueeze(0)

        return out
