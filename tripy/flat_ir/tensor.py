from typing import List
from dataclasses import dataclass

from mlir import ir

from tripy.frontend import Dim
from tripy.common.types import ShapeInfo
from tripy.util import StackInfo, make_list
from tripy.trace.tensor import TraceTensor


@dataclass
class FIRTensor(TraceTensor):
    """
    Represents a single tensor in the FlatIR
    """

    def __init__(self, instance):
        super().__init__(
            instance.name, instance.stack_info, instance.shape, instance.producer, instance.dtype, instance.device
        )

    def to_stablehlo(self):
        from tripy.backend.mlir import utils as mlir_utils

        return ir.RankedTensorType.get(
            [
                ir.ShapedType.get_dynamic_size()
                if (isinstance(s, Dim) and not s.is_static_shape())
                else (s.min if (isinstance(s, Dim)) else s)
                for s in make_list(self.shape)
            ],
            mlir_utils.get_mlir_dtype(self.dtype),
        )
