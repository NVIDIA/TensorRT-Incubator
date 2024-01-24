from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


@dataclass
class Cast(BaseOperator):
    """
    Represents a cast operation.
    """

    to_type: "tripy.common.dtype"

    def infer_shapes(self):
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.to_type

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvertOp

        ConvertOp(self, inputs, outputs)


@TENSOR_METHOD_REGISTRY("float")
def _float(self: "tripy.Tensor"):
    from tripy.frontend import Tensor
    from tripy import float32

    return Tensor.build([self], Cast, float32)
