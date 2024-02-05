from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class MaxOp(BaseFIROp):
    """
    Operation to perform element-wise maximum.
    """

    def to_mlir(self, operands):
        return [stablehlo.maximum(*operands)]
