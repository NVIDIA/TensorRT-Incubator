from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class CompareOp(BaseFlatIROp):
    """
    Operation to compare two tensors
    """

    def __init__(self, origin_layer, inputs, outputs, compare_direction):
        super().__init__(origin_layer, inputs, outputs)
        self.compare_direction = compare_direction

    def to_mlir(self, operands):
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
