from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.common.types import ShapeInfo
from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class IotaOp(BaseFlatIROp):
    """
    Operation to fill an output tensor with values in increasing order starting from zero along the given dimension
    """

    dim: int
    shape: ShapeInfo
    dtype: "tripy.dtype"

    def __init__(self, source_op, inputs, outputs, dim):
        super().__init__(source_op, inputs, outputs)
        assert len(self.outputs) == 1, "IotaOp should have exactly 1 output"
        self.dim = dim
        self.shape = self.outputs[0].shape
        self.dtype = self.outputs[0].dtype

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        iota_dim = ir.IntegerAttr.get(type=ir.IntegerType.get_signless(64), value=self.dim)
        output = stablehlo.IotaOp(out_type, iota_dim)
        return [output]
