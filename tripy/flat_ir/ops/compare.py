from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class CompareOp(BaseFlatIROp):
    compare_direction: str

    def to_mlir(self, operands):
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def _op_name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
