from typing import List
from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class SliceOp(BaseFlatIROp):

    start_indices: List[int]
    limit_indices: List[int]
    strides: List[int]

    def to_mlir(self, operands):
        return [stablehlo.slice(operands[0], self.start_indices, self.limit_indices, self.strides)]
