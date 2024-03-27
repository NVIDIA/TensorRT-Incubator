import dataclasses
from dataclasses import dataclass

from tripy.common.types import ShapeInfo
from tripy.frontend import dynamic_dim
from tripy import utils


@dataclass
class TraceTensor:
    """
    Represents a single tensor in the Trace
    """

    name: str
    """A unique name for the tensor"""

    stack_info: utils.StackInfo
    """Information about the stack where the tensor was created"""

    shape: ShapeInfo
    """Information about the shape of this tensor"""

    producer: "BaseTraceOp"
    """Producer of the tensor"""

    dtype: "tripy.common.dtype"
    """Data type of the tensor"""

    device: "tripy.common.device"
    """Device location of the tensor"""

    def __str__(self) -> str:
        def str_from_dim(dim: dynamic_dim):
            return ("?" if dim.is_dynamic_dim() else str(dim)) + ","

        shape = f"{' '.join(map(str_from_dim, self.shape))}"
        return (
            f"{self.name}: [shape=({shape}), "
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print trace tensors.
        return str(self)

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape

    # Returns a list filled with requested optimization profile information.
    def get_optimization_profile_list(self, attr):
        return [getattr(s, attr) if s.is_dynamic_dim() else s.min for s in utils.make_list(self.shape)]

    def to_flat_ir(self) -> "FlatIRTensor":
        from tripy.flat_ir.tensor import FlatIRTensor

        tensor = FlatIRTensor(**{field.name: getattr(self, field.name) for field in dataclasses.fields(self)})
        # Unset producer to prevent confusing bugs. Otherwise we can end up with FlatIR tensors that point to trace ops.
        tensor.producer = None
        return tensor
