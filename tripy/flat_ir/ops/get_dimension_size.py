from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class GetDimensionSizeOp(BaseFlatIROp):

    dim: int

    def to_mlir(self, operands):
        inp = operands[0]
        dim_attr = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=self.dim,
        )
        return [stablehlo.get_dimension_size(inp, dimension=dim_attr)]
