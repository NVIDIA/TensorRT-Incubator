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

from typing import List
from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo
from mlir_tensorrt.compiler.dialects._ods_common import get_op_result_or_value

from nvtripy.flat_ir.ops.base import BaseFlatIROp
from nvtripy.backend.mlir.utils import is_any_dim_dynamic


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
            assert static_dim_attrs, "DynamicSliceOp requires at-least 1 attribute to be of static shape."
            for d in dynamic_dim_attrs:
                new_shape = [s for s in get_op_result_or_value(static_dim_attrs[0]).type.shape]
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
