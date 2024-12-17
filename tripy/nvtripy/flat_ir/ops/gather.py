#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo
from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value

from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class DynamicGatherOp(BaseFlatIROp):

    axis: int

    def to_mlir(self, operands):
        index_dims = self.inputs[1].rank
        # Ensure slice_sizes is a static tensor with the same shape as the input.
        slice_sizes = get_op_result_or_value(operands[2])
        slice_sizes.set_type(ir.RankedTensorType.get([self.inputs[0].rank], slice_sizes.type.element_type))
        offset_dims = list(range(self.axis)) + list(range(self.axis + index_dims, self.inputs[0].rank + index_dims - 1))
        index_vector_dim = self.inputs[1].rank

        attr = stablehlo.GatherDimensionNumbers.get(
            # The set of dimensions in the output shape that offset into an array sliced from operand.
            offset_dims=offset_dims,
            # The set of dimensions in each slice that are collapsed away. These dimensions must have size 1.
            collapsed_slice_dims=[
                self.axis,
            ],
            operand_batching_dims=[],
            # A map that describes how to map indices in start_indices to legal indices into operand.
            start_index_map=[
                self.axis,
            ],
            # The dimension in start_indices that "contains" the starting indices.
            index_vector_dim=index_vector_dim,
            start_indices_batching_dims=[],
        )

        gather_out = stablehlo.dynamic_gather(
            operand=operands[0], start_indices=operands[1], dimension_numbers=attr, slice_sizes=slice_sizes
        )
        return [gather_out]
