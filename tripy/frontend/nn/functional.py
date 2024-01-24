from tripy.frontend.ops import exp


def softmax(input: "tripy.Tensor", dim: int = None):
    r"""
    Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    Args:
        input: input Tensor
        dim: A dimension along which softmax will be computed.

    Example:
    ::

        a = tp.iota([2, 2], dtype=tp.float32)
        print(f"a: {a}")
        out = tp.nn.functional.softmax(a, dim=1)
        print(f"out: {out}")
        assert np.allclose(out.sum(1).numpy(), np.ones((2), dtype=np.float32))
    """
    exp_inp = exp(input - input.max(dim, keepdim=True))
    return exp_inp / exp_inp.sum(dim, keepdim=True)
