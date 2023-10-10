from dataclasses import dataclass

from tripy.util import StackInfo


@dataclass
class Tensor:
    """
    Represents a single tensor in the FlatIR
    """

    id: int
    """A unique integer ID for the tensor"""

    stack_info: StackInfo
    """Information about the stack where the tensor was created"""
