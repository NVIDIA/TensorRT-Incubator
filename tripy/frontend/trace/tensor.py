from dataclasses import dataclass

import dataclasses
from tripy.common.types import ShapeInfo
from tripy.frontend import Dim
from tripy.util import StackInfo, make_list


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

    producer: "BaseOperator"
    """Producer of the tensor"""

    dtype: "tripy.common.dtype"
    """Data type of the tensor"""

    device: "tripy.common.device"
    """Device location of the tensor"""

    def __str__(self) -> str:
        def str_from_dim(dim: Dim):
            return ("?" if dim.is_dynamic_dim() else str(dim)) + ","

        shape = f"{' '.join(map(str_from_dim, self.shape))}"
        return f"{self.name}: [shape=({shape}), dtype=({self.dtype.name}), loc=({self.device})]"

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape

    # Returns a list filled with requested optimization profile information.
    def get_optimization_profile_list(self, attr):
        return [getattr(s, attr) if s.is_dynamic_dim() else s.min for s in make_list(self.shape)]

    def to_flat_ir(self) -> "FIRTensor":
        from tripy.flat_ir.tensor import FIRTensor

        return FIRTensor(**{field.name: getattr(self, field.name) for field in dataclasses.fields(self)})
