import enum
from dataclasses import dataclass

from tripy.frontend.parameters.base import BaseParameters


@dataclass
class BinaryElementwiseParameters(BaseParameters):
    """
    Represents a binary elementwise operation.
    """

    class Operation(enum.Enum):
        SUM = 0
        """Perform an elementwise sum"""

    operation: Operation
    """The operation to apply"""
