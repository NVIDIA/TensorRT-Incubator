from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConcatenateOp(BaseFlatIROp):
    dim: int

    def to_mlir(self, operands):
        concatenate_dim = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=self.dim,
        )

        output = stablehlo.concatenate(operands, dimension=concatenate_dim)
        return [output]
