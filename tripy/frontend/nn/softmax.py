def softmax(input: "tripy.Tensor", dim: int = None) -> "tripy.Tensor":
    r"""
    Applies the softmax function to the input tensor:

    :math:`\text{softmax}(x_{i}) = \Large \frac{e^{x_{i}}}{\sum_{j=1}^N e^{x_{j}}} \normalsize for\ i=1,2,\dots,N`

    where :math:`x_{i}` is the :math:`i^{th}` element along dimension ``dim``
    and :math:`N` is the size of the dimension.

    Effectively, for each slice along ``dim``, elements are scaled such that they
    lie in the range :math:`[0, 1]` and sum to 1.

    Args:
        input: The input tensor.
        dim: The dimension along which softmax will be computed.
            If this is ``None``, softmax is applied over the whole input array.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota([2, 2], dtype=tp.float32)
        output = tp.nn.softmax(input, dim=0)

        assert np.allclose(output.numpy(), torch.Tensor([[0., 0.], [1., 1.]]).softmax(0).numpy())
    """
    # TODO(#96): make keepdim always True to match the mlir pattern
    exp_inp = (input - input.max(dim, keepdim=True)).exp()
    return exp_inp / exp_inp.sum(dim, keepdim=True)
