from dataclasses import dataclass
from typing import Optional, Sequence, Union

from tripy.common.exception import raise_error
import tripy.frontend.trace.ops.utils as op_utils
from tripy import export, utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Flip(BaseTraceOp):
    dims: Sequence[int]

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import FlipOp

        FlipOp.build([inputs[0]], outputs, dims=self.dims)


@export.public_api(document_under="tensor_operations")
def flip(input: "tripy.Tensor", dims: Optional[Union[int, Sequence[int]]] = None) -> "tripy.Tensor":
    r"""
    Return a new tensor with the same value as the `input` tensor, with the values in the
    dimension(s) given in `dims` reversed. If a value in `dims` is negative,
    it is counted backwards from the last dimension.

    Note that slicing with a negative step size is implemented using `flip`; e.g., `t[::-1]` is
    equivalent to flipping that dimension.

    Args:
        input: The input tensor.
        dims: The dimensions that should be reversed. If `None`, all dimensions will be reversed.
            If a given dimension is negative, it will be counted backwards from the last dimension.

    Returns:
        A new tensor with the same datatype and values as `input`, with the specified dimensions reversed.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(10), (2, 5))
        output = tp.flip(input) # equivalent to tp.flip(input, dims=[0, 1])
        assert cp.array_equal(cp.from_dlpack(output), cp.array([[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]]))

    .. code-block:: python
        :linenos:
        :caption: Reversing only one dimension.

        input = tp.reshape(tp.arange(10), (2, 5))
        output = tp.flip(input, dims=-1)
        assert cp.array_equal(cp.from_dlpack(output), cp.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]))
    """
    rank = input.rank
    if dims is None:
        dims = [d for d in range(rank)]
    else:
        encountered = set()
        dims = utils.make_list(dims)

        if rank == 0 and len(dims) != 0:
            raise_error("It is not possible to flip a rank-0 tensor.")

        for i, dim in enumerate(dims):
            corrected_dim = rank + dim if dim < 0 else dim
            if corrected_dim in encountered:
                dim_message = f"{dim}" if dim >= 0 else f"{corrected_dim} ({dim})"
                raise_error(f"All dimensions for flip must be unique but dimension {dim_message} is repeated.")
            if rank > 0 and (corrected_dim < 0 or corrected_dim >= rank):
                raise_error(
                    f"All dimensions for flip must be in the range [-{rank}, {rank}), but dimension {dim} is out of range."
                )
            dims[i] = corrected_dim
            encountered.add(corrected_dim)

    return Flip.build([input], dims=dims)
