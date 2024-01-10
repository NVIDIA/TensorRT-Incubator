from typing import List

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class AddOp(BaseFIROp):
    """
    Operation to add two tensors
    """

    def __init__(self, origin_layer, inputs, outputs):
        super().__init__(inputs, outputs, origin_layer)

    def to_mlir(self, operands: List) -> List:
        add_out = stablehlo.AddOp(*operands)
        return [add_out]
