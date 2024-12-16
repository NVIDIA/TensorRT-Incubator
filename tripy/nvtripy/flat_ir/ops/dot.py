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
from typing import Dict, List

from mlir_tensorrt.compiler.dialects import stablehlo

from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class DotOp(BaseFlatIROp):

    contracting_dim: Dict[str, List[int]]
    batching_dim: Dict[str, List[int]]

    def to_mlir(self, operands):
        # dot_general spec: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#dot_general
        out_type = self.outputs[0].to_mlir()

        attr = stablehlo.DotDimensionNumbers.get(
            lhs_batching_dimensions=self.batching_dim["lhs"],
            rhs_batching_dimensions=self.batching_dim["rhs"],
            lhs_contracting_dimensions=self.contracting_dim["lhs"],
            rhs_contracting_dimensions=self.contracting_dim["rhs"],
        )

        dot_out = stablehlo.dot_general(result=out_type, lhs=operands[0], rhs=operands[1], dot_dimension_numbers=attr)
        return [dot_out]
