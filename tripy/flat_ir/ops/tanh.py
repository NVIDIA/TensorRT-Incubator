from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


class TanhOp(BaseFlatIROp):
    """
    Operation to perform tanh on a tensor.
    """

    def to_mlir(self, operands):
        return [stablehlo.TanhOp(*operands)]
