from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConcatenateOp(BaseFlatIROp):
    dim: int

    def __init__(self, origin_layer, inputs, outputs, dim):
        super().__init__(origin_layer, inputs, outputs)
        assert len(self.outputs) == 1, "ConcatenateOp should have exactly 1 output"
        self.dim = dim

    def to_mlir(self, operands):
        concatenate_dim = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=self.dim,
        )

        output = stablehlo.concatenate(operands, dimension=concatenate_dim)
        return [output]
