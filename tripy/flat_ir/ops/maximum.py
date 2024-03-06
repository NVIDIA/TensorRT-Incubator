from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class MaxOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [stablehlo.maximum(*operands)]
