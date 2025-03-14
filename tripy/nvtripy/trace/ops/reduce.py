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
from typing import Sequence

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.trace.ops.base import TraceOp


def make_reduce_op(name, attr_name):
    @dataclass(repr=False)
    class ReduceOp(TraceOp):
        dim: Sequence[int]
        keepdim: bool

        def infer_rank(self):
            self.outputs[0].rank = self.inputs[0].rank - (len(self.dim) if not self.keepdim else 0)

        def to_mlir(self, inputs, outputs):
            return [
                tensorrt.reduce(
                    inputs[0],
                    ir.DenseI64ArrayAttr.get(self.dim),
                    tensorrt.ReduceOperationAttr.get(attr_name),
                    keep_dimensions=self.keepdim,
                )
            ]

    ReduceOp.__name__ = name
    return ReduceOp


Sum = make_reduce_op("Sum", "kSUM")
Prod = make_reduce_op("Prod", "kPROD")
Avg = make_reduce_op("Avg", "kAVG")
Max = make_reduce_op("Max", "kMAX")
Min = make_reduce_op("Min", "kMIN")
