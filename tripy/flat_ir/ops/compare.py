from typing import List

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.util.util import get_flat_tensor_info


class CompareOp(BaseFIROp):
    """
    Operation to compare two tensors
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "compare_direction" in kwargs
        self.compare_direction = kwargs.get("compare_direction")

    def add_spaces_around_string(self, s):
        return f" {s} "

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "CompareOp should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__}.{self.compare_direction} {', '.join([f'{get_flat_tensor_info(name, self.inputs[idx])}' for idx, name in enumerate(input_names)])}"

    def to_mlir(self, operands: List) -> List:
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]
