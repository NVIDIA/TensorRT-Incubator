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

from collections.abc import Sequence
from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from nvtripy.trace.ops.base import TraceOp
from mlir_tensorrt.compiler.dialects import tensorrt


@dataclass(repr=False)
class Convolution(TraceOp):
    stride: Sequence[int]
    pre_padding: Sequence[int]
    post_padding: Sequence[int]
    groups: int
    dilation: Sequence[int]

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_mlir(self, inputs, outputs):
        return [
            tensorrt.convolution(
                inputs[0],
                self.stride,
                pre_padding=self.pre_padding,
                post_padding=self.post_padding,
                kernel=inputs[1],
                bias=inputs[2] if len(inputs) > 2 else None,
                num_groups=self.groups,
                dilation=self.dilation,
            )
        ]
