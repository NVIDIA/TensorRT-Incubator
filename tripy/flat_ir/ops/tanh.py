from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class TanhOp(BaseFIROp):
    """
    Operation to perform tanh on a tensor.
    """

    def to_mlir(self, operands: List) -> List:
        return [stablehlo.TanhOp(*operands)]
