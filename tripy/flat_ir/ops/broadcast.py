from mlir import ir
from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.util.util import get_flat_tensor_info


class BroadcastOp(BaseFIROp):
    """
    Operation to expand the dimensions and/or rank of an input tensor by duplicating its data.
    """

    def __init__(self, origin_layer, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, origin_layer)
        assert "broadcast_dim" in kwargs
        self.broadcast_dim = kwargs.get("broadcast_dim")

    def to_flat_ir_str(self, input_names, output_names) -> str:
        assert len(input_names) == 1, "BroadCastOp takes exactly 1 operands"
        return f"{output_names[0]} : {self.__class__.__name__} operand={get_flat_tensor_info(input_names[0], self.inputs[0])}, broadcast_dim={self.broadcast_dim}"

    def to_mlir(self, operands):
        import numpy as np

        out_type = self.outputs[0].to_mlir()
        broadcast_dim_attr = ir.DenseElementsAttr.get(
            np.array(self.broadcast_dim, dtype=np.int64),
            type=ir.IntegerType.get_signless(64),
        )
        output = stablehlo.broadcast_in_dim(out_type, operands[0], broadcast_dim_attr)
        return [output]
