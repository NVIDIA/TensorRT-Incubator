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

from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo
from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value


from nvtripy.flat_ir.ops.base import BaseFlatIROp

# TODO: this should go in top-level Tripy config?
DIM_TENSOR_BITWIDTH = 32


@dataclass(repr=False)
class GetDimensionSizeOp(BaseFlatIROp):

    dim: int

    def to_mlir(self, operands):
        inp = get_op_result_or_value(operands[0])

        inp_type = ir.RankedTensorType(inp.type)
        assert self.dim < inp_type.rank, f"expected dim ({self.dim}) to be less than rank ({inp_type.rank})"
        dim_int_type = ir.IntegerType.get_signless(DIM_TENSOR_BITWIDTH)

        # If we can view the type of the tensor and the dimension is static,
        # then just materialize a constant operation.
        if not ir.ShapedType.is_dynamic_size(inp_type.shape[self.dim]):
            result = stablehlo.constant(
                ir.DenseIntElementsAttr.get_splat(
                    ir.RankedTensorType.get([], dim_int_type),
                    ir.IntegerAttr.get(dim_int_type, inp_type.shape[self.dim]),
                )
            )
            return [result]

        # otherwise, create `stablehlo.get_dimension_size`
        dim_attr = ir.IntegerAttr.get(
            type=ir.IntegerType.get_signless(64),
            value=self.dim,
        )
        result = stablehlo.get_dimension_size(inp, dimension=dim_attr)
        return [result]
