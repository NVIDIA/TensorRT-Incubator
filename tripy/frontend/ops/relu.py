import tripy as tp
from tripy import export


@export.public_api(document_under="tensor_operations")
def relu(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies Rectified Linear Unit (RELU) function
    to each element of the input tensor:

    :math:`\text{relu}(x) = \max(0,x)`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.relu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(cp.from_dlpack(output).get(), np.from_dlpack(torch.nn.functional.relu(t)))

    """
    zeros = tp.zeros((1,), dtype=input.dtype)
    return tp.maximum(zeros, input)
