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

from dataclasses import dataclass
from typing import Optional, Sequence

from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Resize(BaseTraceOp):

    mode: str
    scales: Optional[Sequence[float]]
    align_corners: bool

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ResizeCubicOp, ResizeLinearOp, ResizeNearestOp

        if self.scales:
            from nvtripy.common.datatype import float32, int32
            from nvtripy.flat_ir.ops import ConstantOp, ConvertOp, MulOp
            from nvtripy.flat_ir.tensor import FlatIRTensor

            # construct output_shape using scales
            # inputs[1] is input[0].shape
            # output_shape = (inputs[1].cast(fp32) * scales).cast(int32)
            out_shape = (inputs[0].rank,)
            scales_tensor = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"create scales tensor in resize op."],
            )
            ConstantOp.build([], [scales_tensor], data=self.scales)
            input_shape_f32 = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"convert input shape tensor to float32 in resize op."],
            )
            ConvertOp.build([inputs[1]], [input_shape_f32])
            out_shape_f32 = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=float32,
                device=outputs[0].device,
                reason_details=[f"compute output shape in resize op."],
            )
            MulOp.build([input_shape_f32, scales_tensor], [out_shape_f32])
            out_shape_tensor = FlatIRTensor.build(
                shape=out_shape,
                rank=1,
                dtype=int32,
                device=outputs[0].device,
                reason_details=[f"convert output shape to int32 in resize op."],
            )
            ConvertOp.build([out_shape_f32], [out_shape_tensor])
            inputs[1] = out_shape_tensor

        if self.mode == "nearest":
            ResizeNearestOp.build(inputs, outputs)
        elif self.mode == "cubic":
            ResizeCubicOp.build(inputs, outputs, self.align_corners, cubic_coeff=-0.75)
        else:
            ResizeLinearOp.build(inputs, outputs, self.align_corners)
