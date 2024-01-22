from typing import List
from dataclasses import dataclass

from mlir import ir
from mlir.dialects import stablehlo

from tripy.backend.mlir import utils as mlir_utils
from tripy.flat_ir.ops.base import BaseFIROp
from tripy import int32


@dataclass
class GatherOp(BaseFIROp):
    """
    Operation to compute gather operation.
    """

    offset_dims: tuple
    axis: int
    slice_sizes: list
    index_vector_dim: int

    def __init__(self, origin_layer, inputs, outputs, axis):
        super().__init__(origin_layer, inputs, outputs)
        index_dims = len(inputs[1].shape)
        self.axis = axis
        self.slice_sizes = [s.runtime_value for s in inputs[0].shape]
        self.slice_sizes[self.axis] = 1

        self.offset_dims = tuple(
            list(range(axis)) + list(range(axis + index_dims, len(inputs[0].shape) + index_dims - 1))
        )
        self.index_vector_dim = len(inputs[1].shape)

    def to_mlir(self, operands: List) -> List:
        attr = stablehlo.GatherDimensionNumbers.get(
            # The set of dimensions in the output shape that offset into an array sliced from operand.
            offset_dims=self.offset_dims,
            # The set of dimensions in each slice that are collapsed away. These dimensions must have size 1.
            collapsed_slice_dims=(self.axis,),
            # A map that describes how to map indices in start_indices to legal indices into operand.
            start_index_map=(self.axis,),
            # The dimension in start_indices that "contains" the starting indices. See below for a detailed description.
            index_vector_dim=self.index_vector_dim,
        )

        gather_out = stablehlo.gather(
            operand=operands[0], start_indices=operands[1], dimension_numbers=attr, slice_sizes=tuple(self.slice_sizes)
        )
        return [gather_out]
