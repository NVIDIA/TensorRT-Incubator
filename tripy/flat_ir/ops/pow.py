from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class PowOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [stablehlo.PowOp(*operands)]
