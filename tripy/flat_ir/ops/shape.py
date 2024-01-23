from mlir import ir

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.backend.mlir import utils as mlir_utils
from tripy import int32


class ShapeOp(BaseFIROp):
    """
    Operation to transpose/permute a Tensor
    """

    def to_mlir(self, operands):
        out_type = ir.RankedTensorType.get([self.outputs[0].shape[0].runtime_value], mlir_utils.get_mlir_dtype(int32))
        inp = operands[0]
        if not isinstance(operands[0], ir.OpResult):
            inp = inp.result

        # Remove use of tensorrt dialect and use shape dialect. #80 will fix this.
        out_shape = ir.Operation.create("tensorrt.shape", results=[out_type], operands=[inp]).result
        return [out_shape]
