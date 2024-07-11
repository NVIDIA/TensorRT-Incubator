from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp

from tripy.common.datatype import float64, float32, float16, bfloat16, uint8, int8, int16, int32, int64, bool as tp_bool


@dataclass(repr=False)
class ConvertOp(BaseFlatIROp):
    def to_mlir(self, operands):
        output = stablehlo.ConvertOp(result=self.outputs[0].to_mlir(), operand=operands[0])
        return [output]
