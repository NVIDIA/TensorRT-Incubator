from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class AddOp(BaseFlatIROp):
    """
    Operation to add two tensors
    """

    def to_mlir(self, operands):
        add_out = stablehlo.AddOp(*operands)
        return [add_out]
