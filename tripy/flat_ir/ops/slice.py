from typing import List
from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from tripy.flat_ir.ops.base import BaseFlatIROp
from tripy.backend.mlir.utils import is_any_dim_dynamic
import tripy.utils.utils as utils


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

        attrs = {
            operands[1]: is_any_dim_dynamic(operands[1]),
            operands[2]: is_any_dim_dynamic(operands[2]),
            operands[3]: is_any_dim_dynamic(operands[3]),
        }

        dynamic_dim_attrs = [v for v, is_dyanmic in attrs.items() if is_dyanmic]
        static_dim_attrs = [v for v, is_dyanmic in attrs.items() if not is_dyanmic]

        if any(dynamic_dim_attrs):
            assert len(static_dim_attrs) > 1, "DynamicSliceOp requires at-least 1 attribute to be of static shape."
            for d in dynamic_dim_attrs:
                new_shape = [s for s in static_dim_attrs[0].type.shape]
                d.set_type(ir.RankedTensorType.get(new_shape, d.type.element_type))

        return [
            stablehlo.real_dynamic_slice(
                result=self.outputs[0].to_mlir(),
                operand=operands[0],
                start_indices=operands[1],
                limit_indices=operands[2],
                strides=operands[3],
            )
        ]
