import math


def gelu(input: "tripy.Tensor"):
    r"""
    Applies a gelu function.

    Gelu is defined as:

    :math:`\text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))`

    Performed on a tensor elementwise.

    Args:
        input: input Tensor

    Example:

    .. code:: python
        :number-lines:

        a = tp.iota([3, 3], dtype=tp.float32)
        out = tp.nn.gelu(a)
        print(f"out: {out}")
        np_a = a.numpy() # doc: omit
        np_out = 0.5 * np_a * (1 + np.tanh(np.sqrt(2 / np.pi) * (np_a + 0.044715 * np.power(np_a, 3)))) # doc: omit
        assert np.allclose(out.numpy(), np_out)
    """
    from tripy.frontend.tensor import Tensor

    # Gelu constants.
    vars = [0.5, math.sqrt(2.0 / math.pi), 0.044715, 3.0, 1.0]

    # Cast to tensors.
    t1, t2, t3, t4, t5 = [Tensor(var) for var in vars]

    return t1 * input * ((t2 * (input + t3 * (input**t4))).tanh() + t5)
