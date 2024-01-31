from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class Shape(BaseOperator):
    """
    Represents a shape operation.
    """

    def infer_shapes(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        from tripy.frontend import Dim

        self.outputs[0].shape = (Dim(len(self.inputs[0].shape)),)

    def infer_dtypes(self):
        from tripy import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ShapeOp

        ShapeOp(self, inputs, outputs)


@TENSOR_METHOD_REGISTRY("shape")
@property
def shape(self) -> "tripy.Tensor":
    """
    Represents the shape of the tensor.

    Returns:
        A 1D tensor containing the shape of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((8, 2))
        shape = input.shape

        assert np.array_equal(shape.numpy(), np.array([8, 2]))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Shape)
