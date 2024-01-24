from tripy.frontend.nn.module import Module
from tripy.frontend.nn.parameter import Parameter


class LayerNorm(Module):
    """
    Applies layer normalization operation.

    Args:
        normalized_shape (int): The feature dimension of the input over which normalization is performed.

    Example:
    ::

        import torch # doc: omit
        a = tp.ones((2, 3))
        ln = tp.nn.LayerNorm(3)
        out = ln(a)

        torch_tensor = torch.ones((2,3), dtype=torch.float32) # doc: omit
        layer_norm = torch.nn.LayerNorm(3) # doc: omit

        # Set weights and biases to 1
        layer_norm.weight.data.fill_(1) # doc: omit
        layer_norm.bias.data.fill_(1) # doc: omit

        print(out)
        assert out.numpy().shape == (2, 3)
        assert np.array_equal(out.numpy(), layer_norm(torch_tensor).detach().numpy())
    """

    def __init__(self, normalized_shape):
        super().__init__()
        from tripy.common.datatype import float32
        from tripy.frontend.tensor_ops import ones

        # Replace with random weights when #74 is completed.
        self.weight = Parameter(ones((normalized_shape,), dtype=float32))
        self.bias = Parameter(ones((normalized_shape,), dtype=float32))
        self.eps = 1e-5

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True) + self.eps
        x = (x - mean) * var.rsqrt()
        return self.weight * x + self.bias
