from tripy import export


@export.public_api(document_under="tensor_operations")
def sigmoid(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies a logistic sigmoid function to each element of the input tensor:

    :math:`\text{sigmoid}(x)_i = \frac{1}{1 + \exp{-x_i}}`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.sigmoid(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(cp.from_dlpack(output).get(), np.from_dlpack(torch.nn.functional.sigmoid(t)))
    """
    from tripy.frontend.trace.ops.unary_elementwise import exp

    return 1.0 / (1.0 + exp(-1.0 * input))