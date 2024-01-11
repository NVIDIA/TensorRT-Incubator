from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


class BroadcastOp(BaseFIROp):
    """
    Operation to expand the dimensions and/or rank of an input tensor by duplicating its data.
    """

    def __init__(self, origin_layer, inputs, outputs, broadcast_dim):
        super().__init__(origin_layer, inputs, outputs)
        self.broadcast_dim = broadcast_dim

    def to_flat_ir_str(self) -> str:
        outputs_str = f"{str(self.outputs[0])}"
        return f"{outputs_str} = {self.name()}({', '.join(list(map(str, self.inputs)))}, broadcast_dim={self.broadcast_dim})"

    def to_mlir(self, operands):
        import numpy as np

        out_type = self.outputs[0].to_mlir()
        broadcast_dim_attr = ir.DenseElementsAttr.get(
            np.array(self.broadcast_dim, dtype=np.int64),
            type=ir.IntegerType.get_signless(64),
        )
        output = stablehlo.broadcast_in_dim(out_type, operands[0], broadcast_dim_attr)
        return [output]
