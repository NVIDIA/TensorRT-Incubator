from typing import List
from dataclasses import dataclass

from mlir import ir

from tripy.frontend import Dim
from tripy.common.types import ShapeInfo
from tripy.util import StackInfo, make_list
from tripy.flat_ir.flat_ir import FlatIR


@dataclass
class TraceTensor:
    """
    Represents a single tensor in the Trace
    """

    name: str
    """A unique name for the tensor"""

    stack_info: StackInfo
    """Information about the stack where the tensor was created"""

    shape: ShapeInfo
    """Information about the shape of this tensor"""

    producer: "TraceLayer"
    """Producer of the tensor"""

    dtype: "tripy.common.dtype"
    """Data type of the tensor"""

    device: "tripy.common.device"
    """Device location of the tensor"""

    def __str__(self) -> str:
        return f"{self.name} : shape=({self.shape}), dtype=({self.dtype.name}), loc=({self.device.kind}:{self.device.index})"

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape

    # Returns a list filled with requested optimization profile information.
    def get_optimization_profile_list(self, attr):
        return [
            getattr(s, attr)
            if (isinstance(s, Dim) and not s.is_static_shape())
            else (s.min if (isinstance(s, Dim)) else s)
            for s in make_list(self.shape)
        ]

    def to_mlir(self):
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
