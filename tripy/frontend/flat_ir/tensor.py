from dataclasses import dataclass

from tripy.util import StackInfo


@dataclass
class Tensor:
    """
    Represents a single tensor in the FlatIR
    """

    name: str
    """A unique name for the tensor"""

    stack_info: StackInfo
    """Information about the stack where the tensor was created"""

    def __str__(self) -> str:
        return self.name
