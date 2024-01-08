from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class SelectOp(BaseFIROp):
    """
    Operation to select values from either x or y, depending on condition.
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert len(inputs) == 3, "SelectOp takes exactly 3 operands"

    def to_flat_ir_str(self, input_names, output_names) -> str:
        return f"{output_names[0]} : {self.__class__.__name__} condition={input_names[0]}, x={input_names[1]}, y={input_names[2]}"

    def to_mlir(self, operands):
        return [stablehlo.select(*operands)]
