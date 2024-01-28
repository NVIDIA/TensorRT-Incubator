from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


class LayerNorm(Module):
    r"""
    Applies layer normalization over the input tensor:

    :math:`\text{LayerNorm}(x) = \Large \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} \normalsize * \gamma + \beta`

    Args:
        normalized_shape: The size of the feature dimension of the input over which normalization is performed.

    Example:

    .. code:: python
        :number-lines:

        layer_norm = tp.nn.LayerNorm(3)

        input = tp.ones((2, 3))
        output = layer_norm(input)

        np_out = output.numpy() # doc: omit
        assert np_out.shape == (2, 3)

        torch_tensor = torch.ones((2,3), dtype=torch.float32) # doc: omit
        layer_norm = torch.nn.LayerNorm(3) # doc: omit
        layer_norm.weight.data.fill_(1) # doc: omit
        layer_norm.bias.data.fill_(1) # doc: omit
        assert np.array_equal(np_out, layer_norm(torch_tensor).detach().numpy())
    """

    def __init__(self, normalized_shape: int):
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.ops import ones

        # Replace with random weights when #74 is completed.
        self.weight: Parameter = Parameter(ones((normalized_shape,), dtype=float32))
        r"""The :math:`\gamma` parameter of shape :math:`[\text{normalized_shape}]`."""

        self.bias: Parameter = Parameter(ones((normalized_shape,), dtype=float32))
        r"""The :math:`\beta` parameter of shape :math:`[\text{normalized_shape}]`."""

        self.eps: float = 1e-5
        """A value added to the denominator to prevent division by zero."""

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor.

        Returns:
            A tensor of the same shape as the input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True) + self.eps
        x = (x - mean) * var.rsqrt()
        return self.weight * x + self.bias
