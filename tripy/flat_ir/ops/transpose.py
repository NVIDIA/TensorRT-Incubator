from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class TransposeOp(BaseFIROp):
    """
    Operation to transpose/permute a Tensor
    """

    def __init__(self, origin_layer, inputs, outputs, perm):
        super().__init__(origin_layer, inputs, outputs)
        self.perm = perm

    def to_flat_ir_str(self) -> str:
        return super().to_flat_ir_str() + f"perm={self.perm}"

    def to_mlir(self, operands):
        output = stablehlo.TransposeOp(operands[0], self.perm)
        return [output]
