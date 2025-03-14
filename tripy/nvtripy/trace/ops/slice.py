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
class Slice(TraceOp):

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_mlir(self, inputs, outputs):
        assert len(inputs) == 4, "Slice operation must have exactly 4 inputs."

        return [tensorrt.slice(inputs[0], start=inputs[1], size=inputs[2], stride=inputs[3])]


@dataclass(repr=False)
class SliceFill(TraceOp):

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_mlir(self, inputs, outputs):
        assert len(inputs) == 5, "SliceFill operation must have exactly 5 inputs."

        mode_attr = tensorrt.SliceModeAttr.get("kFILL")

        return [
            tensorrt.slice(
                inputs[0],
                start=inputs[1],
                size=inputs[2],
                stride=inputs[3],
                fill=inputs[4],
                mode=mode_attr,
            )
        ]
