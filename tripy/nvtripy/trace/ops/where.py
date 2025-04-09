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
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class Where(TraceOp):
    infer_rank = op_utils.InferRankPolicies.max_of_inputs()

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        self.outputs[0].dtype = self.inputs[1].dtype

    def to_mlir(self, inputs, outputs):
        return [tensorrt.select(inputs[0], inputs[1], inputs[2])]
