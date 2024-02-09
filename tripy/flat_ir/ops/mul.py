from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class MulOp(BaseFlatIROp):
    """
    Operation to multiply two tensors
    """

    def to_mlir(self, operands):
        return [stablehlo.MulOp(*operands)]
