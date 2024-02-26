from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class SubtractOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.subtract(*operands)]
