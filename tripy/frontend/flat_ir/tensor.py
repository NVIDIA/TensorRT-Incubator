from dataclasses import dataclass

from tripy.util import StackInfo

ShapeInfo = tuple[int, ...]


@dataclass
class Tensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    """A unique name for the tensor"""

    stack_info: StackInfo
    """Information about the stack where the tensor was created"""

    shape: ShapeInfo
    """Information about the shape of this tensor"""

    def __str__(self) -> str:
        return f"{self.name} [{self.shape}]"
