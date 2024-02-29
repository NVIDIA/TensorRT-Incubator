from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class CompareOp(BaseFlatIROp):
    compare_direction: str

    def to_mlir(self, operands):
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
