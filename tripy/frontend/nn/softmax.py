def softmax(input: "tripy.Tensor", dim: int = None):
    r"""
    Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1. If dim is None, softmax is applied to the
    entire input array.

    Args:
        input: input Tensor
        dim: A dimension along which softmax will be computed.

    Example:

    .. code:: python
        :number-lines:

        a = tp.iota([2, 2], dtype=tp.float32)
        print(f"a: {a}")
        out = tp.nn.softmax(a, dim=0)
        print(f"out: {out}")
        assert np.allclose(out.numpy(), torch.Tensor([[0., 0.], [1., 1.]]).softmax(0).numpy())
    """
    # TODO(#96): make keepdim always True to match the mlir pattern
    exp_inp = (input - input.max(dim, keepdim=True)).exp()
    return exp_inp / exp_inp.sum(dim, keepdim=True)
