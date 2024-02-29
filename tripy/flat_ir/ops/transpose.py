from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class TransposeOp(BaseFlatIROp):
    perm: List[int]

    def to_mlir(self, operands):
        output = stablehlo.TransposeOp(operands[0], self.perm)
        return [output]
