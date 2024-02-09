from dataclasses import dataclass
from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class TransposeOp(BaseFlatIROp):
    """
    Operation to transpose/permute a Tensor
    """

    perm: List[int]

    def __init__(self, origin_layer, inputs, outputs, perm):
        super().__init__(origin_layer, inputs, outputs)
        self.perm = perm

    def to_mlir(self, operands):
        output = stablehlo.TransposeOp(operands[0], self.perm)
        return [output]
