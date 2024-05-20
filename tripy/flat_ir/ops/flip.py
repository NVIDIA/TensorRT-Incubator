from dataclasses import dataclass
from typing import Sequence

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class FlipOp(BaseFlatIROp):
    dims: Sequence[int]

    def to_mlir(self, operands):
        dims_attr = ir.DenseI64ArrayAttr.get(self.dims)
        return [stablehlo.reverse(operands[0], dims_attr)]
