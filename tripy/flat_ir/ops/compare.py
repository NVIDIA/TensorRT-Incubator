from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(init=False, repr=False)
class CompareOp(BaseFlatIROp):
    def __init__(self, source_op, inputs, outputs, compare_direction):
        super().__init__(source_op, inputs, outputs)
        self.compare_direction = compare_direction

    def to_mlir(self, operands):
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
