from nvtripy import export
from nvtripy.common import datatype
from nvtripy.frontend.ops.full import full_like
from nvtripy.frontend.ops.iota import iota_like
from nvtripy.frontend.ops.zeros import zeros_like
from nvtripy.frontend.ops.where import where
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"tensor": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int32", "int64", "bool"],
    },
)
def tril(tensor: "nvtripy.Tensor", diagonal: int = 0) -> "nvtripy.Tensor":
    r"""
    Returns the lower triangular part of each :math:`[M, N]` matrix in the tensor, with all other elements set to 0.
    If the tensor has more than two dimensions, it is treated as a batch of matrices.

    Args:
        tensor: The nvtripy tensor to operate on.
        diagonal: The diagonal above which to zero elements.
            ``diagonal=0`` indicates the main diagonal which is defined by the set of indices
            :math:`{{(i, i)}}` where :math:`i \in [0, min(M, N))`.

            Positive values indicate the diagonal which is that many diagonals above the main one,
            while negative values indicate one which is below.

    Returns:
        A tensor of the same shape as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Main Diagonal

        input = tp.iota((2, 1, 3, 3), dim=2) + 1.
        output = tp.tril(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.tril(cp.from_dlpack(input).get()))

    .. code-block:: python
        :linenos:
        :caption: Two Diagonals Above Main

        input = tp.iota((5, 5)) + 1. # doc: omit
        output = tp.tril(input, diagonal=2)

        assert np.array_equal(cp.from_dlpack(output).get(), np.tril(cp.from_dlpack(input).get(), 2))

    .. code-block:: python
        :linenos:
        :caption: One Diagonal Below Main

        input = tp.iota((5, 5)) + 1. # doc: omit
        output = tp.tril(input, diagonal=-1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.tril(cp.from_dlpack(input).get(), -1))
    """
    tri_mask = (iota_like(tensor, -2, datatype.int32) + full_like(tensor, diagonal, datatype.int32)) >= iota_like(
        tensor, -1, datatype.int32
    )
    zeros_tensor = zeros_like(tensor)
    return where(tri_mask, tensor, zeros_tensor)
