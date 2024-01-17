from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class ShapeOf(BaseOperator):
    """
    Represents a shape operation.
    """

    def infer_shapes(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        from tripy.frontend import Dim

        self.outputs[0].shape = (Dim(len(self.inputs[0].shape)),)

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("cpu")

    def infer_dtypes(self):
        from tripy import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import ShapeOfOp

        flat_ir.add_op(self, ShapeOfOp, self.inputs, self.outputs)


@TENSOR_METHOD_REGISTRY("shape")
@property
def shape(self: "tripy.Tensor"):
    """
    Returns the shape of the tensor.

    Returns:
        1d tensor filled with shape of the tensor.

    Example:
    ::

        input = tp.ones((8, 2))
        print(input.shape)
        assert np.array_equal(input.shape.numpy(), np.array([8, 2]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], ShapeOf)
