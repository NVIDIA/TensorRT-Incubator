import math

from tripy import export


@export.public_api(document_under="tensor_operations")
def gelu(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies an approximated Gaussian Error Linear Units (GELU) function
    to each element of the input tensor:

    :math:`\text{gelu}(x) = 0.5 * x * (1 + \tanh(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.gelu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(output.numpy(), torch.nn.functional.gelu(t, approximate='tanh').numpy())
    """
    from tripy.frontend.trace.ops.unary_elementwise import tanh

    t1, t2, t3, t4, t5 = 0.5, math.sqrt(2.0 / math.pi), 0.044715, 3.0, 1.0
    return t1 * input * (tanh(t2 * (input + t3 * (input**t4))) + t5)
