from dataclasses import dataclass
from typing import List

from tripy.flat_ir.tensor import Tensor
from tripy.frontend.parameters import BaseParameters


@dataclass
class Layer:
    """
    Represents a single layer in the FlatIR
    """

    inputs: List[Tensor]
    """The inputs of this layer"""

    output: Tensor
    """The output of this layer"""

    params: BaseParameters
    """Parameters describing the operation of this layer"""
