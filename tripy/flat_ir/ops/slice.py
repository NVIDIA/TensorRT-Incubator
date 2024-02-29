from typing import List
from dataclasses import dataclass

from mlir.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class SliceOp(BaseFlatIROp):
    """
    Operation to slice a tensor.
    """

    start_indices: List[int]
    limit_indices: List[int]
    strides: List[int]

    def __init__(self, source_op, inputs, outputs, start_indices, limit_indices, strides):
        super().__init__(source_op, inputs, outputs)
        assert len(inputs) == 1, "SliceOp takes exactly one operand"
        assert len(start_indices) == len(limit_indices) and len(start_indices) == len(strides)
        self.start_indices = start_indices
        self.limit_indices = limit_indices
        self.strides = strides

    def to_mlir(self, operands):
        return [stablehlo.slice(operands[0], self.start_indices, self.limit_indices, self.strides)]


@dataclass(repr=False)
class DynamicSliceOp(BaseFlatIROp):
    """
    Operation to dynamically slice a tensor.
    """

    def __init__(self, source_op, inputs, outputs):
        super().__init__(source_op, inputs, outputs)
        assert len(inputs) == 4, "DynamicSliceOp takes exactly one operand"

    def to_mlir(self, operands):

        return [
            stablehlo.real_dynamic_slice(
                result=self.outputs[0].to_mlir(),
                operand=operands[0],
                start_indices=operands[1],
                limit_indices=operands[2],
                strides=operands[3],
            )
        ]
