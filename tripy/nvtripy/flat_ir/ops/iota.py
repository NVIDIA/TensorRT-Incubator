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

from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class DynamicIotaOp(BaseFlatIROp):

    dim: int

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        iota_dim = ir.IntegerAttr.get(type=ir.IntegerType.get_signless(64), value=self.dim)
        output = stablehlo.DynamicIotaOp(result=out_type, output_shape=operands[0], iota_dimension=iota_dim)
        return [output]
