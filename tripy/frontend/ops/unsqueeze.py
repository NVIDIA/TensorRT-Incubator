from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
import tripy.frontend.ops.utils as op_utils


@dataclass
class Unsqueeze(BaseOperator):
    """
    Represents an unsqueeze operation.
    """

    dim: int

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Unsqueeze operation should have exactly one input!"
        if self.dim < 0:
            self.dim += len(self.inputs[0].shape) + 1

        out_shape = list(self.inputs[0].shape)
        out_shape.insert(self.dim, 1)
        self.outputs[0].shape = op_utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import BroadcastOp

        broadcast_dim = list(range(len(inputs[0].shape)))
        for idx in range(len(broadcast_dim)):
            if idx >= self.dim:
                broadcast_dim[idx] += 1
        BroadcastOp(self, inputs, outputs, broadcast_dim=broadcast_dim)


@TENSOR_METHOD_REGISTRY("unsqueeze")
def unsqueeze(self: "tripy.Tensor", dim: int):
    """
    Returns a new tensor with a dimension of size one inserted at the specified position.

    Args:
        dim: the index at which to insert the singleton dimension

    Returns:
        the unsqueezed Tensor

    Example:

    .. code:: python

        input = tp.ones((2, 2), dtype=tp.float32)
        output = input.unsqueeze(1)

        assert np.array_equal(output.numpy(), np.expand_dims(input.numpy(), 1))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Unsqueeze, dim)
