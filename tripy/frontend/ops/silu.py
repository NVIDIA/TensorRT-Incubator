from tripy import export


@export.public_api(document_under="tensor_operations")
def silu(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function  to each element of the
    input tensor. This function is also known as the swish function.

    :math:`\text{silu}(x) = x \cdot \sigma (x)`
    where
    :math:`\sigma (x)_i = \frac{1}{1 + \exp{-x_i}}`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.silu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(cp.from_dlpack(output).get(), np.from_dlpack(torch.nn.functional.silu(t)))
    """
    from tripy.frontend.trace.ops.unary_elementwise import exp

    return input / (1.0 + exp(-1.0 * input))