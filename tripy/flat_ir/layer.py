from dataclasses import dataclass
from typing import List

from tripy.flat_ir.tensor import FIRTensor
from tripy.ops import BaseOperator


@dataclass
class FIRLayer:
    """
    Represents a single layer in the FlatIR
    """

    inputs: List[FIRTensor]
    """The inputs of this layer"""

    outputs: List[FIRTensor]
    """The outputs of this layer"""

    op: BaseOperator
    """The operation applied by this layer"""
