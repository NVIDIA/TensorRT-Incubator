from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class SelectOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [stablehlo.select(*operands)]
