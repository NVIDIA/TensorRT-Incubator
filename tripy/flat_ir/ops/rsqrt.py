from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(init=False, repr=False)
class RsqrtOp(BaseFlatIROp):

    def to_mlir(self, operands):
        return [stablehlo.RsqrtOp(*operands)]
