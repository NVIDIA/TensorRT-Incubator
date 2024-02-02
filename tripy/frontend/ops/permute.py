from dataclasses import dataclass
from typing import Sequence

import tripy.frontend.ops.utils as op_utils
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass(repr=False)
class Permute(BaseOperator):
    """
    Represents a permute operation.
    """

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
                    f"elements as the rank of the input tensor (shape: {input_shape}, rank: {len(input_shape)})."
                ],
            )

        self.outputs[0].shape = tuple(input_shape[idx] for idx in self.permutation)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import TransposeOp

        TransposeOp(self, inputs, outputs, perm=self.permutation)


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


@TENSOR_METHOD_REGISTRY("transpose")
def transpose(self, dim0: int, dim1: int) -> "tripy.Tensor":
    """
    Returns a new tensor that is a transposed version of this one where
    ``dim0`` and ``dim1`` are swapped.

    Args:
        dim0: The first dimension to be transposed.
        dim1: The second dimension to be transposed.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.transpose(0, 1)

        assert np.array_equal(output.numpy(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Transpose, None, dim0, dim1)


@TENSOR_METHOD_REGISTRY("permute")
def permute(self, perm: Sequence[int]) -> "tripy.Tensor":
    """
    Returns a tensor with its dimensions permuted.

    Args:
        perm: The desired ordering of dimensions.
              It must contain all integers in :math:`[0..N-1]` exactly once,
              where :math:`N` is the rank of this tensor.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.arange(6, dtype=tp.float32).reshape((2, 3))
        output = input.permute((1, 0))

        assert np.array_equal(output.numpy(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Permute, perm)
