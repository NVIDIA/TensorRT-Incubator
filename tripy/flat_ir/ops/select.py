from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class SelectOp(BaseFlatIROp):
    """
    Operation to select values from either x or y, depending on condition.
    """

    def to_mlir(self, operands):
        return [stablehlo.select(*operands)]
