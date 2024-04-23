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
    stack_info: utils.StackInfo
    shape: ShapeInfo
    dtype: "tripy.common.dtype"
    device: "tripy.common.device"
    producer: "BaseTraceOp"
    rank: int

    def __str__(self) -> str:
        def str_from_dim(dim: dynamic_dim):
            return ("?" if dim.is_dynamic_dim() else str(dim)) + ","

        shape = f"{' '.join(map(str_from_dim, self.shape))}"
        return (
            f"{self.name}: [shape=({shape}), "
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"rank=({self.rank}), "
            + f"loc=({self.device})]"
        )

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print trace tensors.
        return str(self)

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape

    def to_flat_ir(self) -> "FlatIRTensor":
        from tripy.flat_ir.tensor import FlatIRTensor

        tensor = FlatIRTensor(
            name=self.name, stack_info=self.stack_info, shape=self.shape, dtype=self.dtype, device=self.device
        )
        return tensor
