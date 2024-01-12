from dataclasses import dataclass
from typing import Sequence

from tripy import utils
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.common.exception import TripyException
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.utils import to_dims


@dataclass
class Reshape(BaseOperator):
    """
    Represents a reshape operation.
    """

    shape: Sequence[int]

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Reshape operation should have exactly one input!"
        self.outputs[0].shape = to_dims(self.shape)

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import ReshapeOp

        if any(
            (dim[0].is_dynamic_dim() or dim[1].is_dynamic_dim())
            for dim in zip(self.inputs[0].shape, to_dims(self.shape))
        ):
            raise NotImplementedError("Dynamic reshape is not supported")

        flat_ir.add_op(self, ReshapeOp, self.inputs, self.outputs)


@TENSOR_METHOD_REGISTRY("reshape")
def reshape(self: "tripy.Tensor", shape: ShapeInfo):
    """
    Reshapes the input tensor into the the given shape.

    Args:
        shape: the new requested shape of tensor

    Returns:
        the reshaped Tensor

    Example:
    ::

        import numpy as np

        t = np.random.rand(2, 4, 4, 6).astype(np.float32)
        a = tp.Tensor(t)
        out = a.reshape((2, 4, 2, 12))
        assert (out.numpy() == np.reshape(t, (2, 4, 2, 12))).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Reshape, shape)
