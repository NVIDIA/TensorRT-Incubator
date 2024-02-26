from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class MaxOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.maximum(*operands)]
