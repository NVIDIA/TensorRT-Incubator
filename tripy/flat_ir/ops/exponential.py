from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class ExpOp(BaseFIROp):
    """
    Operation to calculate exponential values of a tensor.
    """

    def to_mlir(self, operands: List) -> List:
        return [stablehlo.exponential(operands[0])]
