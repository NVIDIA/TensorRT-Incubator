from typing import List
from dataclasses import dataclass

from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class SliceOp(BaseFlatIROp):

    start_indices: List[int]
    limit_indices: List[int]
    strides: List[int]

    def to_mlir(self, operands):
        return [stablehlo.slice(operands[0], self.start_indices, self.limit_indices, self.strides)]


@dataclass(repr=False)
class DynamicSliceOp(BaseFlatIROp):
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
