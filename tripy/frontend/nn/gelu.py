import math


def gelu(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies an approximated Gaussian Error Linear Units (GELU) function
    to each element of the input tensor:

    :math:`\text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))`

    Args:
        input: The input tensor.

    Returns:
        An output tensor of the same size as the input.

    Example:

    .. code:: python
        :number-lines:

        a = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        out = tp.nn.gelu(a)
        print(f"out: {out}")

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(out.numpy(), torch.nn.functional.gelu(t, approximate='tanh').numpy())
    """
    t1, t2, t3, t4, t5 = 0.5, math.sqrt(2.0 / math.pi), 0.044715, 3.0, 1.0
    return t1 * input * ((t2 * (input + t3 * (input**t4))).tanh() + t5)
