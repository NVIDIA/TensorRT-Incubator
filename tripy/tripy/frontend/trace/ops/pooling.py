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

import enum
from dataclasses import dataclass
from typing import Sequence, Tuple
from tripy import constraints

import tripy.frontend.trace.ops.utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Pooling(BaseTraceOp):

    class Kind(enum.Enum):
        def __init__(self, op):
            self.op = op

        MAX = "max"
        AVG = "avg"

    kind: Kind
    kernel_dims: Sequence[int]
    stride: Sequence[int]
    padding: Sequence[Tuple[int]]

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConstantOp, ReduceWindowOp
        from tripy.flat_ir.tensor import FlatIRTensor

        init_value = 0
        init_const = FlatIRTensor.build(
            shape=(),
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[
                f"create the constant value tensor (containing {init_value}) for the initial value of a '{self.kind.op}' operation"
            ],
        )
        ConstantOp.build([], [init_const], data=init_value)

        # extend parameters [spatial_dims,] -> [rank(input),]
        extra_dims = inputs[0].rank - len(self.kernel_dims)
        window_dims = [1] * extra_dims + list(self.kernel_dims)
        window_strides = [1] * extra_dims + list(self.stride)
        padding = [(0, 0)] * extra_dims + list(self.padding)

        ReduceWindowOp.build(
            [inputs[0], init_const],
            outputs,
            reduce_mode=self.kind.op,
            window_dims=window_dims,
            window_strides=window_strides,
            padding=padding,
        )


@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8"],
    },
    dtype_constraints={"input": "T1", "weight": "T1", constraints.RETURN_VALUE: "T1"},
)
def maxpool(
    input: "tripy.Tensor",
    kernel_dims: Sequence[int],
    stride: Sequence[int],
    padding: Sequence[Tuple[int]],
):
    return Pooling.build([input], Pooling.Kind.MAX, kernel_dims, stride, padding)
