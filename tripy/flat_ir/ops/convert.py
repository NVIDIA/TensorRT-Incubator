from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConvertOp(BaseFlatIROp):
    def to_mlir(self, operands):
        output = stablehlo.ConvertOp(result=self.outputs[0].to_mlir(), operand=operands[0])
        return [output]
