from dataclasses import dataclass

from mlir import ir

from tripy.frontend.trace.tensor import TraceTensor
from tripy import utils
from tripy.common.types import ShapeInfo


@dataclass
class FIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in utils.make_list(self.shape)],
            mlir_utils.get_mlir_dtype(self.dtype),
        )

    @staticmethod
    def build(shape: ShapeInfo, dtype: "tripy.common.dtype", device: "tripy.common.device") -> "FIRTensor":
        return FIRTensor(
            name=None, stack_info=utils.get_stack_info(), producer=None, shape=shape, dtype=dtype, device=device
        )
