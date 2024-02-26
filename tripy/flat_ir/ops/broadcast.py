from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class BroadcastOp(BaseFlatIROp):
    broadcast_dim: int

    def __init__(self, source_op, inputs, outputs, broadcast_dim):
        super().__init__(source_op, inputs, outputs)
        self.broadcast_dim = broadcast_dim

    def to_mlir(self, operands):
        import numpy as np

        out_type = self.outputs[0].to_mlir()
        broadcast_dim_attr = ir.DenseElementsAttr.get(
            np.array(self.broadcast_dim, dtype=np.int64),
            type=ir.IntegerType.get_signless(64),
        )
        output = stablehlo.broadcast_in_dim(out_type, operands[0], broadcast_dim_attr)
        return [output]


@dataclass(repr=False)
class DynamicBroadcastOp(BaseFlatIROp):

    broadcast_dim: int

    def __init__(self, source_op, inputs, outputs, broadcast_dim):
        super().__init__(source_op, inputs, outputs)
        self.broadcast_dim = broadcast_dim

    def to_mlir(self, operands):
        import numpy as np

        broadcast_dim_attr = ir.DenseI64ArrayAttr.get(
            np.array(self.broadcast_dim, dtype=np.int64),
        )
        out_type = self.outputs[0].to_mlir()

        output = stablehlo.dynamic_broadcast_in_dim(out_type, operands[0], operands[1], broadcast_dim_attr)
        return [output]
