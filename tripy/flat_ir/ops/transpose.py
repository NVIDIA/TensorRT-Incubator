from dataclasses import dataclass
from typing import List

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class TransposeOp(BaseFlatIROp):
    perm: List[int]

    def to_mlir(self, operands):
        perm_attr = ir.DenseI64ArrayAttr.get(self.perm)
        output = stablehlo.TransposeOp(operands[0], perm_attr)
        return [output]
