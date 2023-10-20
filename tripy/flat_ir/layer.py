from dataclasses import dataclass
from typing import List

from tripy.flat_ir.tensor import Tensor
from tripy.ops import BaseOperator


@dataclass
class Layer:
    """
    Represents a single layer in the FlatIR
    """

    inputs: List[Tensor]
    """The inputs of this layer"""

    outputs: List[Tensor]
    """The outputs of this layer"""

    op: BaseOperator
    """The operation applied by this layer"""
