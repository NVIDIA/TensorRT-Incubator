from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class IotaOp(BaseFlatIROp):

    dim: int

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        iota_dim = ir.IntegerAttr.get(type=ir.IntegerType.get_signless(64), value=self.dim)
        output = stablehlo.IotaOp(out_type, iota_dim)
        return [output]


@dataclass(repr=False)
class DynamicIotaOp(BaseFlatIROp):

    dim: int

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        iota_dim = ir.IntegerAttr.get(type=ir.IntegerType.get_signless(64), value=self.dim)
        output = stablehlo.DynamicIotaOp(result=out_type, output_shape=operands[0], iota_dimension=iota_dim)
        return [output]
