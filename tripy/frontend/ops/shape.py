from dataclasses import dataclass
from typing import Sequence

from tripy import utils
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.common.exception import TripyException


@dataclass
class ShapeOf(BaseOperator):
    """
    Represents a shape operation.
    """

    dim: int

    def to_trace_str(self):
        return f"{self.outputs[0].name} = Tensor.shape({self.inputs[0].name}, dim={self.dim})"

    def infer_shapes(self):
        assert len(self.inputs) == 1, "ShapeOf operation should have exactly one input!"
        from tripy.frontend import Dim

        self.outputs[0].shape = (Dim(len(self.inputs[0].shape)),)

    def infer_dtypes(self):
        from tripy import int32

        self.outputs[0].dtype = int32

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import ShapeOfOp

        flat_ir.add_op(self, ShapeOfOp, self.inputs, self.outputs, dim=self.dim)


@TENSOR_METHOD_REGISTRY("shape")
def shape(self: "tripy.Tensor", dim=None):
    """
    Returns the shape of the tensor. If dim is specified, returns the shape an integer holding the shape at dim.
    Args:
        dim0 (Optional): The dimension along which shape is requested.

    Returns:
        1d tensor filled with

    Example:
    ::

        import numpy as np

        input = tp.ones((128, 20))
        assert (input.shape().numpy() == np.array([128, 20])).all()
    """
    from tripy.frontend import Tensor

    assert dim is None, "Gather op required to index into shape tensor not implemented."

    return Tensor.build([self], ShapeOf, dim)
