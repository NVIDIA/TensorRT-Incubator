from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class MinOp(BaseFlatIROp):
    def to_mlir(self, operands):
        return [stablehlo.minimum(*operands)]
