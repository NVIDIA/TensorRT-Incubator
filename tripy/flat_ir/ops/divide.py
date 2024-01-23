from typing import List

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class DivideOp(BaseFIROp):
    """
    Operation to divide a tensor.
    """

    def to_mlir(self, operands):
        return [stablehlo.divide(*operands)]
