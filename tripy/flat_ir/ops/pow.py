from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class PowOp(BaseFlatIROp):
    """
    Operation to perform element-wise exponentiation
    """

    def to_mlir(self, operands):
        return [stablehlo.PowOp(*operands)]
