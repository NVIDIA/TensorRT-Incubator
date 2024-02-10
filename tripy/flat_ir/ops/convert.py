from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ConvertOp(BaseFlatIROp):
    """
    Operation to cast a Tensor to output type
    """

    def __init__(self, source_op, inputs, outputs):
        super().__init__(source_op, inputs, outputs)

    def to_mlir(self, operands):
        output = stablehlo.ConvertOp(result=self.outputs[0].to_mlir(), operand=operands[0])
        return [output]
