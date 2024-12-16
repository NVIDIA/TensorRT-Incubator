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
from typing import List

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import stablehlo

from nvtripy.backend.mlir.utils import is_any_dim_dynamic
from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class DynamicBroadcastOp(BaseFlatIROp):

    broadcast_dim: List[int]

    def to_mlir(self, operands):

        if is_any_dim_dynamic(operands[1]):
            # Tripy frontend does not have shape inference and stablehlo does not allow shape operand to be of dynamic shape.
            # Since DynamicBroadcastOp was created internally by Tripy, we know the expected output rank. For dynamic_broadcast_in_dim operator, the shape of shape tensor is the same as output rank.
            new_shape = [self.outputs[0].rank]
            self.inputs[1].shape = new_shape
            operands[1].set_type(ir.RankedTensorType.get(new_shape, operands[1].type.element_type))

        broadcast_dim_attr = ir.DenseI64ArrayAttr.get(self.broadcast_dim)
        out_type = self.outputs[0].to_mlir()

        output = stablehlo.dynamic_broadcast_in_dim(out_type, operands[0], operands[1], broadcast_dim_attr)
        return [output]
