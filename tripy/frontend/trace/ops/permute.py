from dataclasses import dataclass
from typing import Sequence

import tripy.frontend.trace.ops.utils as op_utils
from tripy import export
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Permute(BaseTraceOp):
    permutation: Sequence[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Permute operation should have exactly one input!"
        input_shape = self.inputs[0].shape

        if len(self.permutation) != len(input_shape):
            op_utils.raise_error_io_info(
                self,
                "Incorrect number of elements in permutation.",
                details=[
                    f"In operation: 'permute', permutation was: {self.permutation}, containing "
                    f"{len(self.permutation)} element(s), but it must have the same number of "
                    f"elements as the rank of the input tensor (shape: {input_shape}, rank: {len(input_shape)})"
                ],
            )

        self.outputs[0].shape = tuple(input_shape[idx] for idx in self.permutation)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import TransposeOp

        TransposeOp.build(inputs, outputs, perm=self.permutation)


@dataclass(repr=False)
class Transpose(Permute):
    """
    Represents a transpose operation.
    """

    dim0: int
    dim1: int

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Transpose operation should have exactly one input!"
        input_shape = self.inputs[0].shape

        perm = list(range(len(input_shape)))
        perm[self.dim0], perm[self.dim1] = perm[self.dim1], perm[self.dim0]
        self.permutation = perm

        return super().infer_shapes()


@export.public_api(document_under="tensor_operations")
def transpose(input: "tripy.Tensor", dim0: int, dim1: int) -> "tripy.Tensor":
    """
    Returns a new tensor that is a transposed version of the input tensor where
    ``dim0`` and ``dim1`` are swapped.

    Args:
        input: The input tensor.
        dim0: The first dimension to be transposed.
        dim1: The second dimension to be transposed.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.transpose(input, 0, 1)

        assert np.array_equal(output.numpy(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Transpose, None, dim0, dim1)


@export.public_api(document_under="tensor_operations")
def permute(input: "tripy.Tensor", perm: Sequence[int]) -> "tripy.Tensor":
    """
    Returns a tensor with its dimensions permuted.

    Args:
        input: The input tensor.
        perm: The desired ordering of dimensions.
              It must contain all integers in :math:`[0..N-1]` exactly once,
              where :math:`N` is the rank of the input tensor.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.permute(input, (1, 0))

        assert np.array_equal(output.numpy(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Permute, perm)
