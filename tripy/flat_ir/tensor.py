from dataclasses import dataclass

from tripy import utils
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.tensor import TraceTensor


@dataclass(repr=False)
class FIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return mlir_utils.make_mlir_tensor(self.shape, self.dtype)

    @staticmethod
    def build(shape: ShapeInfo, dtype: "tripy.common.dtype", device: "tripy.common.device") -> "FIRTensor":
        return FIRTensor(
            name=None, stack_info=utils.get_stack_info(), producer=None, shape=shape, dtype=dtype, device=device
        )
