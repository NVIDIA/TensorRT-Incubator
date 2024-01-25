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
def expand(self: "tripy.Tensor", sizes: Sequence[int]):
    """
    Returns a new view of the self tensor with singleton dimensions expanded to a larger size.

    Passing -1 as the size for a dimension means not changing the size of that dimension.

    Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front.

    Args:
        sizes: the desired expanded size

    Returns:
        the expanded Tensor

    Example:

    .. code:: python
        :number-lines:

        a = tp.ones((2, 1), dtype=tp.float32)
        out = a.expand((2, -1, 4))
        print(out)
        assert np.array_equal(out.numpy(), np.broadcast_to(np.ones((2, 1), dtype=np.float32), (2, 2, 4)))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Expand, sizes)
