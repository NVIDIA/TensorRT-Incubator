from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class CompareOp(BaseFIROp):
    """
    Operation to compare two tensors
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "compare_direction" in kwargs
        self.compare_direction = kwargs.get("compare_direction")

    def to_mlir(self, operands: List) -> List:
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
