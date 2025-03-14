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


def make_unary_op(name, attr_name):
    @dataclass(repr=False)
    class UnaryOp(TraceOp):
        infer_rank = op_utils.InferRankPolicies.same_as_input()

        def to_mlir(self, inputs, outputs):
            return [tensorrt.unary(inputs[0], tensorrt.UnaryOperationAttr.get(attr_name))]

    UnaryOp.__name__ = name
    return UnaryOp


Exp = make_unary_op("Exp", "kEXP")
Recip = make_unary_op("Recip", "kRECIP")
Log = make_unary_op("Log", "kLOG")
Sin = make_unary_op("Sin", "kSIN")
Cos = make_unary_op("Cos", "kCOS")
Sqrt = make_unary_op("Sqrt", "kSQRT")
Abs = make_unary_op("Abs", "kABS")
Not = make_unary_op("Not", "kNOT")
Neg = make_unary_op("Neg", "kNEG")


def make_activation_op(name, attr_name):
    @dataclass(repr=False)
    class ActivationOp(TraceOp):
        infer_rank = op_utils.InferRankPolicies.same_as_input()

        def to_mlir(self, inputs, outputs):
            return [tensorrt.activation(inputs[0], tensorrt.ActivationTypeAttr.get(attr_name))]

    ActivationOp.__name__ = name
    return ActivationOp


Tanh = make_activation_op("Tanh", "kTANH")
Relu = make_activation_op("Relu", "kRELU")
Sigmoid = make_activation_op("Sigmoid", "kSIGMOID")
GeluErf = make_activation_op("GeluErf", "kGELU_ERF")
