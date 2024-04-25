from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ReshapeOp(BaseFlatIROp):
    def to_mlir(self, operands):
        output = stablehlo.ReshapeOp(result=self.outputs[0].to_mlir(), operand=operands[0])
        return [output]


class DynamicReshapeOp(BaseFlatIROp):
    def to_mlir(self, operands):
        output = stablehlo.dynamic_reshape(
            result=self.outputs[0].to_mlir(), operand=operands[0], output_shape=operands[1]
        )
        return [output]
