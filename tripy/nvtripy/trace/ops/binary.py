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
from dataclasses import dataclass

import nvtripy.trace.ops.utils as op_utils
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.common import datatype
from nvtripy.trace.ops.base import TraceOp


def make_binary_op(name, attr_name):
    @dataclass(repr=False)
    class BinaryOp(TraceOp):
        def infer_rank(self):
            assert self.inputs[0].rank == self.inputs[1].rank, "All inputs must have the same rank"
            return op_utils.InferRankPolicies.same_as_input()(self)

        def to_mlir(self, inputs, outputs):
            return [tensorrt.element_wise(inputs[0], inputs[1], tensorrt.ElementWiseOperationAttr.get(attr_name))]

    BinaryOp.__name__ = name
    return BinaryOp


Add = make_binary_op("Add", "kSUM")
Sub = make_binary_op("Sub", "kSUB")
Pow = make_binary_op("Pow", "kPOW")
Mul = make_binary_op("Mul", "kPROD")
Div = make_binary_op("Div", "kDIV")
FloorDiv = make_binary_op("FloorDiv", "kFLOOR_DIV")
Max = make_binary_op("Max", "kMAX")
Min = make_binary_op("Min", "kMIN")


def make_binary_comparison_op(name, attr_name):
    @dataclass(repr=False)
    class BinaryComparisonOp(make_binary_op(name, attr_name)):
        def infer_dtypes(self):
            self.outputs[0].dtype = datatype.bool

    BinaryComparisonOp.__name__ = name
    return BinaryComparisonOp


Less = make_binary_comparison_op("Less", "kLESS")
Greater = make_binary_comparison_op("Greater", "kGREATER")
Equal = make_binary_comparison_op("Equal", "kEQUAL")
LogicalOr = make_binary_comparison_op("LogicalOr", "kOR")
