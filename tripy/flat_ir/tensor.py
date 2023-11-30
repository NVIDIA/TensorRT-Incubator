from typing import List
from dataclasses import dataclass

from mlir import ir

from tripy.common.types import ShapeInfo
from tripy.util import StackInfo, make_list


@dataclass
class FIRTensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    """A unique name for the tensor"""

    stack_info: StackInfo
    """Information about the stack where the tensor was created"""

    shape: ShapeInfo
    """Information about the shape of this tensor"""

    producer: "FIRLayer"
    """Producer of the tensor"""

    dtype: "tripy.frontend.DataType"
    """Data type of the tensor"""

    def __str__(self) -> str:
        return f"{self.name} [{self.shape}]"

    def __eq__(self, other: "FIRTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape

    def to_mlir(self):
        from tripy.backend.mlir import utils as mlir_utils

        return ir.RankedTensorType.get(
            [ir.ShapedType.get_dynamic_size() if s == -1 else s for s in make_list(self.shape)],
            mlir_utils.convert_dtype(self.dtype),
        )
