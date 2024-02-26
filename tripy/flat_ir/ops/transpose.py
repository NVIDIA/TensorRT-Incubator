from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class TransposeOp(BaseFlatIROp):
    perm: List[int]

    def __init__(self, source_op, inputs, outputs, perm):
        super().__init__(source_op, inputs, outputs)
        self.perm = perm

    def to_mlir(self, operands):
        output = stablehlo.TransposeOp(operands[0], self.perm)
        return [output]
