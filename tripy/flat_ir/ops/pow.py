from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class PowOp(BaseFIROp):
    """
    Operation to perform element-wise exponentiation
    """

    def to_mlir(self, operands: List) -> List:
        add_out = stablehlo.PowOp(*operands)
        return [add_out]
