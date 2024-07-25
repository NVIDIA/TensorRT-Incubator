from dataclasses import dataclass

from tripy import utils
from typing import Optional, List


@dataclass
class TraceTensor:
    """
    Represents a single tensor in the Trace
    """

    name: str
    stack_info: utils.StackInfo
    dtype: "tripy.common.dtype"
    device: "tripy.common.device"
    rank: int
    producer: "BaseTraceOp"
    shape: Optional[List[int]] = None

    def __str__(self) -> str:
        return (
            f"{self.name}: [rank=({self.rank}), "
            + (f"shape=({self.shape}), " if self.shape is not None else "")
            + (f"dtype=({self.dtype.name}), " if self.dtype is not None else "")
            + f"loc=({self.device})]"
        )

    def __repr__(self) -> str:
        # This is a hack to prevent printing the entire stack info when we print trace tensors.
        return str(self)

    def __eq__(self, other: "TraceTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info

    def to_flat_ir(self) -> "FlatIRTensor":
        from tripy.flat_ir.tensor import FlatIRTensor

        tensor = FlatIRTensor(
            name=self.name,
            stack_info=self.stack_info,
            dtype=self.dtype,
            device=self.device,
            rank=self.rank,
            shape=self.shape,
        )
        return tensor
