from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class ExpOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.exponential(operands[0])]
