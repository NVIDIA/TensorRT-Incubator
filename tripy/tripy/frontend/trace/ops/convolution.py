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

from collections.abc import Sequence
from dataclasses import dataclass
from tripy import constraints

import tripy.frontend.trace.ops.utils as op_utils
from tripy import utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Convolution(BaseTraceOp):
    padding: Sequence[Sequence[int]]
    stride: Sequence[int]
    groups: int
    lhs_dilation: Sequence[int]
    rhs_dilation: Sequence[int]

    def verify_spatial_rank(self, attr, rank, string):
        spatial_rank = rank - 2
        if attr and len(attr) != spatial_rank:
            utils.raise_error_io_info(
                self,
                f"Number of {string} values does not match number of spatial dimensions in the input.",
                details=[
                    f"Got {len(attr)} {string} value pairs but the number of spatial dimensions is: {spatial_rank}.",
                ],
            )

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def validate_inputs(self, tensor_shape, kernel_shape):
        if len(tensor_shape) != len(kernel_shape):
            utils.raise_error_io_info(
                self,
                "Input tensor and kernel must have the same rank.",
                details=[
                    f"Input tensor for operation: 'convolution' has shape: {tensor_shape} [rank = {len(tensor_shape)}], "
                    f"but should have the same rank as the kernel of shape: {kernel_shape} [rank = {len(kernel_shape)}]."
                ],
            )

        rank = len(tensor_shape)

        self.verify_spatial_rank(self.padding, rank, "padding")
        self.verify_spatial_rank(self.stride, rank, "stride")
        self.verify_spatial_rank(self.lhs_dilation, rank, "lhs_dilation")
        self.verify_spatial_rank(self.rhs_dilation, rank, "rhs_dilation")

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import ConvolutionOp

        ConvolutionOp.build(
            inputs,
            outputs,
            padding=self.padding,
            stride=self.stride,
            feature_group_count=self.groups,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.rhs_dilation,
        )


@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", "weight": "T1", constraints.RETURN_VALUE: "T1"},
)
def convolution(
    input: "tripy.Tensor",
    weight: "tripy.Tensor",
    padding: Sequence[Sequence[int]],
    stride: Sequence[int],
    groups: int,
    lhs_dilation: Sequence[int],
    rhs_dilation: Sequence[int],
):
    return Convolution.build([input, weight], padding, stride, groups, lhs_dilation, rhs_dilation)
