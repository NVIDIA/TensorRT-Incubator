from typing import List
from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFIROp


@dataclass(repr=False)
class SliceOp(BaseFIROp):
    """
    Operation to slice a tensor.
    """

    start_indices: List[int]
    limit_indices: List[int]
    strides: List[int]

    def __init__(self, origin_layer, inputs, outputs, start_indices, limit_indices, strides):
        super().__init__(origin_layer, inputs, outputs)
        assert len(inputs) == 1, "SliceOp takes exactly one operand"
        assert len(start_indices) == len(limit_indices) and len(start_indices) == len(strides)
        self.start_indices = start_indices
        self.limit_indices = limit_indices
        self.strides = strides

    def to_mlir(self, operands):
        return [stablehlo.slice(operands[0], self.start_indices, self.limit_indices, self.strides)]
