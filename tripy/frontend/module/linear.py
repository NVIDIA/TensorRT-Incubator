from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter
from tripy.utils import export


@export.public_api(document_under="modules")
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

        linear = tp.Linear(3, 4)

        input = tp.iota((2, 3))
        output = linear(input)

        assert output.numpy().shape == (2, 4)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype: datatype = datatype.float32):
        super().__init__()
        from tripy.frontend.ops import ones

        self.dtype = dtype
        r"""The dtype to perform the Linear operation"""

        # Replace with random weights when #74 is completed.
        self.weight: Parameter = Parameter(ones((out_features, in_features), dtype=dtype))
        r"""The :math:`W` matrix of shape :math:`[\text{out_features}, \text{in_features}]`"""

        if bias:
            self.bias: Parameter = Parameter(ones((out_features), dtype=dtype))
            r"""The :math:`b` matrix of shape :math:`[1, \text{out_features}]`"""

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor, of shape :math:`[*, \text{in_features}]`.

        Returns:
            A tensor of shape :math:`[*, \text{out_features}]`.
        """
        from tripy.frontend.trace.ops.permute import transpose
        from tripy.frontend.trace.ops.unsqueeze import unsqueeze

        out = x @ (transpose(self.weight, 1, 0))
        if hasattr(self, "bias"):
            out = out + unsqueeze(self.bias, 0)

        return out
