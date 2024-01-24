from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


@dataclass
class ConvertOp(BaseFIROp):
    """
    Operation to cast a Tensor to output type
    """

    def __init__(self, origin_layer, inputs, outputs):
        super().__init__(origin_layer, inputs, outputs)

    def to_mlir(self, operands):
        output = stablehlo.ConvertOp(result=self.outputs[0].to_mlir(), operand=operands[0])
        return [output]
