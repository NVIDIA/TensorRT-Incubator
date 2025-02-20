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

from mlir_tensorrt.compiler.dialects import stablehlo

from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class CompareOp(BaseFlatIROp):
    compare_direction: str

    def to_mlir(self, operands):
        compare_out = stablehlo.CompareOp(*operands, stablehlo.ComparisonDirectionAttr.get(self.compare_direction))
        return [compare_out]

    def _op_name(self) -> str:
        return f"{self.__class__.__name__}.{self.compare_direction}"
