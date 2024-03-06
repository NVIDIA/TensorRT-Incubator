from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(init=False, repr=False)
class LogOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [stablehlo.log(*operands)]
