from dataclasses import dataclass
from typing import Sequence

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
import tripy.frontend.ops.utils as op_utils


@dataclass
class Expand(BaseOperator):
    """
    Represents an expand operation.
    """

    shape: Sequence[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Expand operation should have exactly one input!"

        input_shape = self.inputs[0].shape
        input_rank = len(input_shape)
        if len(self.shape) < input_rank:
            op_utils.raise_error_io_info(
                self,
                "The number of sizes must be greater or equal to input tensor's rank.",
                [f"Target sizes: {self.shape}", f" input rank: {input_rank}"],
            )

        extra_dims = len(self.shape) - input_rank
        out_shape = list(self.shape[:extra_dims])
        for idx, s in enumerate(self.shape[extra_dims:]):
            if s == -1 or s == input_shape[idx].runtime_value:
                out_shape.append(input_shape[idx])
            elif input_shape[idx].runtime_value == 1:
                out_shape.append(s)
            else:
                op_utils.raise_error_io_info(
                    self,
                    f"The expanded size must match the existing size at non-singleton dimension.",
                    [f"Expanded size at index {idx}: {s}", f" existing non-singleton size: {input_shape[idx]}"],
                )

        self.outputs[0].shape = op_utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import BroadcastOp

        broadcast_dim = op_utils.get_broadcast_in_dim(inputs[0].shape, outputs[0].shape)
        BroadcastOp(self, inputs, outputs, broadcast_dim=broadcast_dim)


@TENSOR_METHOD_REGISTRY("expand")
def expand(self, sizes: Sequence[int]) -> "tripy.Tensor":
    """
    Returns a new tensor based on this tensor with singleton dimensions expanded to a larger size.

    Args:
        sizes: The desired expanded size.
            A value of :math:`-1` indicates that the dimension should not be modified.
            If the length of this parameter exceeds the rank of the tensor, new dimensions
            are prepended.

    Returns:
        The new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((2, 1), dtype=tp.float32)
        output = input.expand((-1, 4))

        assert np.array_equal(output.numpy(), np.broadcast_to(np.ones((2, 1), dtype=np.float32), (2, 4)))

    .. code-block:: python
        :linenos:
        :caption: Increasing Tensor Rank

        input = tp.ones((1, 1), dtype=tp.float32)
        output = input.expand((3, -1, -1))

        assert np.array_equal(output.numpy(), np.broadcast_to(np.ones((1, 1), dtype=np.float32), (3, 1, 1)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Expand, sizes)
