from typing import List

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.util.util import get_flat_tensor_info


class AddOp(BaseFIROp):
    """
    Operation to add two tensors
    """

    def __init__(self, origin_layer, inputs, outputs):
        super().__init__(inputs, outputs, origin_layer)

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(output_names) == 1, "AddOp should have exactly one output!"
        return f"{output_names[0]} = {self.__class__.__name__} {', '.join([f'{get_flat_tensor_info(name, self.inputs[idx])}' for idx, name in enumerate(input_names)])}"

    def to_mlir(self, operands: List) -> List:
        add_out = stablehlo.AddOp(*operands)
        return [add_out]
