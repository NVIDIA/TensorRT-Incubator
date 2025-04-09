#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class MatrixMultiply(TraceOp):
    def infer_rank(self):
        out_rank = max(self.inputs[0].rank, self.inputs[1].rank)
        if self.inputs[0].rank == 1 and self.inputs[1].rank == 1:
            out_rank = 0
        elif self.inputs[0].rank == 1 or self.inputs[1].rank == 1:
            out_rank = 1
        self.outputs[0].rank = out_rank

    def to_mlir(self, inputs, outputs):
        lhs_rank = inputs[0].type.rank
        rhs_rank = inputs[1].type.rank
        # This should be based on the tensor ranks.
        lhs_op = tensorrt.MatrixOperationAttr.get("kNONE" if lhs_rank > 1 else "kVECTOR")
        rhs_op = tensorrt.MatrixOperationAttr.get("kNONE" if rhs_rank > 1 else "kVECTOR")
        return [tensorrt.matrix_multiply(inputs[0], inputs[1], lhs_op, rhs_op)]
