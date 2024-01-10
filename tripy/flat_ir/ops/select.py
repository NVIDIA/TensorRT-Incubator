from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class SelectOp(BaseFIROp):
    """
    Operation to select values from either x or y, depending on condition.
    """

    def __init__(self, origin_layer, inputs, outputs):
        super().__init__(origin_layer, inputs, outputs)
        assert len(inputs) == 3, "SelectOp takes exactly 3 operands"

    def to_mlir(self, operands):
        return [stablehlo.select(*operands)]
