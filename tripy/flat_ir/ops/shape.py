from mlir import ir

from tripy.flat_ir.ops.base import BaseFIROp
from tripy.backend.mlir import utils as mlir_utils
from tripy import int32


class ShapeOfOp(BaseFIROp):
    """
    Operation to transpose/permute a Tensor
    """

    def to_mlir(self, operands):
        out_type = ir.RankedTensorType.get([self.outputs[0].shape[0].runtime_value], mlir_utils.get_mlir_dtype(int32))
        out_shape = ir.Operation.create("tensorrt.shape", results=[out_type], operands=[operands[0].result]).result
        return [out_shape]
