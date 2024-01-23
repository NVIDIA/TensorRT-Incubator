from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class SubtractOp(BaseFIROp):
    """
    Operation to subtract one tensor from another.
    """

    def to_mlir(self, operands):
        return [stablehlo.subtract(*operands)]
