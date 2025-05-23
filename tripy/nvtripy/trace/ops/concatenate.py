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

import nvtripy.trace.ops.utils as op_utils
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class Concatenate(TraceOp):
    dim: int

    def infer_rank(self):
        assert len(set(inp.rank for inp in self.inputs)) == 1, "All inputs must have the same rank!"
        return op_utils.InferRankPolicies.same_as_input()(self)

    def to_mlir(self, inputs, outputs):
        output = tensorrt.concatenation(inputs, axis=self.dim)
        return [output]
