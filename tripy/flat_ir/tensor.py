from dataclasses import dataclass

from mlir import ir

from tripy.frontend.trace.tensor import TraceTensor
from tripy.utils import make_list


@dataclass
class FIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if s.is_dynamic_dim() else s.min for s in make_list(self.shape)],
            mlir_utils.get_mlir_dtype(self.dtype),
        )
