from typing import List
from dataclasses import dataclass

from tripy.types import ShapeInfo
from tripy.util import StackInfo


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

    def __str__(self) -> str:
        return f"{self.name} [{self.shape}]"

    def __eq__(self, other: "FIRTensor") -> bool:
        return self.name == other.name and self.stack_info == other.stack_info and self.shape == other.shape
