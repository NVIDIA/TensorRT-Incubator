from dataclasses import dataclass
from typing import List

from tripy.trace.tensor import TraceTensor
from tripy.ops import BaseOperator


@dataclass
class TraceLayer:
    """
    Represents a single layer in the Trace
    """

    inputs: List[TraceTensor]
    """The inputs of this layer"""

    outputs: List[TraceTensor]
    """The outputs of this layer"""

    op: BaseOperator
    """The operation applied by this layer"""
