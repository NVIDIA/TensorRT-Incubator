from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class MulOp(BaseFIROp):
    """
    Operation to multiply two tensors
    """

    def to_mlir(self, operands: List) -> List:
        return [stablehlo.MulOp(*operands)]
