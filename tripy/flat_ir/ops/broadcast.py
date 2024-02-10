from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class BroadcastOp(BaseFlatIROp):
    """
    Operation to expand the dimensions and/or rank of an input tensor by duplicating its data.
    """

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
    """
    Dynamic shape variant of the BroadcastOp.
    """

    broadcast_dim: int

    def __init__(self, source_op, inputs, outputs, broadcast_dim):
        super().__init__(source_op, inputs, outputs)
        self.broadcast_dim = broadcast_dim

    def to_mlir(self, operands):
        import numpy as np

        broadcast_dim_attr = ir.DenseI64ArrayAttr.get(
            np.array(self.broadcast_dim, dtype=np.int64),
        )

        # Use tensorrt dialect until https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/635 is resolved.
        out_shape = ir.Operation.create(
            "tensorrt.broadcast",
            results=[self.outputs[0].to_mlir()],
            operands=operands,
            attributes={"broadcast_dims": broadcast_dim_attr},
        ).result
        return [out_shape]
