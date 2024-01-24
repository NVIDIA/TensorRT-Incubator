from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class RsqrtOp(BaseFIROp):
    """
    Operation to perform rsqrt on a tensor.
    """

    def to_mlir(self, operands):
        return [stablehlo.RsqrtOp(*operands)]
