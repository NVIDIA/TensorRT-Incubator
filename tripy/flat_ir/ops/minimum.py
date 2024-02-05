from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class MinOp(BaseFIROp):
    """
    Operation to perform element-wise minimum.
    """

    def to_mlir(self, operands):
        return [stablehlo.minimum(*operands)]
