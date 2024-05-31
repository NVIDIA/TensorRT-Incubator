from dataclasses import dataclass
from typing import Sequence

import tripy.frontend.trace.ops.utils as op_utils
from tripy import export, utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Expand(BaseTraceOp):
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

        self.outputs[0].shape = utils.to_dims(out_shape)

    def infer_rank(self):
        self.outputs[0].rank = len(self.shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import BroadcastOp

        broadcast_dim = op_utils.get_broadcast_in_dim(inputs[0].rank, outputs[0].rank)
        BroadcastOp.build(inputs, outputs, broadcast_dim=broadcast_dim)


@export.public_api(document_under="tensor_operations")
def expand(input: "tripy.Tensor", sizes: Sequence[int]) -> "tripy.Tensor":
    """
    Returns a new tensor based on the input tensor with singleton dimensions expanded to a larger size.

    Args:
        input: The input tensor.
        sizes: The desired expanded size.
            A value of :math:`-1` indicates that the dimension should not be modified.
            If the length of this parameter exceeds the rank of the tensor, new dimensions
            are prepended.

    Returns:
        The new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 1), dtype=tp.float32)
        output = tp.expand(input, (-1, 4))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (2, 4)))

    .. code-block:: python
        :linenos:
        :caption: Increasing Tensor Rank

        input = tp.iota((1, 1), dtype=tp.float32)
        output = tp.expand(input, (3, -1, -1))

        assert np.array_equal(cp.from_dlpack(output).get(), np.broadcast_to(cp.from_dlpack(input).get(), (3, 1, 1)))
    """
    return Expand.build([input], sizes)
