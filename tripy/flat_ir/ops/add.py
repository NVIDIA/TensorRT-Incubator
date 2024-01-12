from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class AddOp(BaseFIROp):
    """
    Operation to add two tensors
    """

    def to_mlir(self, operands: List) -> List:
        add_out = stablehlo.AddOp(*operands)
        return [add_out]
