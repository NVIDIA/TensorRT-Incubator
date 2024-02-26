from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class SelectOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.select(*operands)]
