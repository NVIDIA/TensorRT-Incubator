from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(init=False, repr=False)
class MulOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.MulOp(*operands)]
