from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class ExpOp(BaseFlatIROp):
    """
    Operation to calculate exponential values of a tensor.
    """

    def to_mlir(self, operands):
        return [stablehlo.exponential(operands[0])]
