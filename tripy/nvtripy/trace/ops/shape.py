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
from nvtripy.common.datatype import int32
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class Shape(TraceOp):
    def infer_rank(self):
        # The shape of shape tensors is always known:
        self.outputs[0].shape = (self.inputs[0].rank,)

    def infer_dtypes(self):
        self.outputs[0].dtype = int32

    def to_mlir(self, inputs, outputs):
        return [tensorrt.shape(inputs[0])]


# This is a special case of slice that is only designed to get a single element from a shape.
@dataclass(repr=False)
class GetDimensionSize(TraceOp):
    dim: int

    def infer_rank(self):
        self.outputs[0].rank = 0

    def infer_dtypes(self):
        self.outputs[0].dtype = int32

    def to_mlir(self, inputs, outputs):
        return [
            tensorrt.collapse_rank(
                outputs[0],
                tensorrt.slice(
                    inputs[0],
                    static_start=[self.dim],
                    static_size=[1],
                    static_stride=[1],
                ),
            )
        ]
