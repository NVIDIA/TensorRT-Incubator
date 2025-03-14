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
from typing import Optional, Sequence

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class ResizeBase(TraceOp):

    scales: Optional[Sequence[float]]

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def get_scales_and_shape(self, inputs):
        assert len(inputs) == 1 or len(inputs) == 2, "Resize must have exactly 1 or 2 inputs."
        if len(inputs) == 2:
            output_shape = inputs[1]
            scales_attr = None
        else:
            assert self.scales, "Resize scales must be provided when there is only 1 input."
            output_shape = None
            scales_attr = ir.DenseF32ArrayAttr.get(self.scales)
        return scales_attr, output_shape


@dataclass(repr=False)
class ResizeNearest(ResizeBase):

    def to_mlir(self, inputs, outputs):
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        scales_attr, output_shape = self.get_scales_and_shape(inputs)
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get("kASYMMETRIC")
        rounding_mode_attr = tensorrt.ResizeRoundModeAttr.get("kFLOOR")
        return [
            tensorrt.resize_nearest(
                outputs[0],
                inputs[0],
                coord_trans_attr,
                rounding_mode_attr,
                selector_attr,
                output_shape=output_shape,
                scales=scales_attr,
            )
        ]


@dataclass(repr=False)
class ResizeCubic(ResizeBase):

    align_corners: bool

    def to_mlir(self, inputs, outputs):
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        scales_attr, output_shape = self.get_scales_and_shape(inputs)
        cubic_coeff = -0.75
        coord_trans = "kALIGN_CORNERS" if self.align_corners else "kHALF_PIXEL"
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get(coord_trans)
        cubic_coeff_attr = ir.FloatAttr.get(ir.F32Type.get(), cubic_coeff)
        return [
            tensorrt.resize_cubic(
                outputs[0],
                inputs[0],
                coord_trans_attr,
                selector_attr,
                cubic_coeff_attr,
                output_shape=output_shape,
                scales=scales_attr,
            )
        ]


@dataclass(repr=False)
class ResizeLinear(ResizeBase):

    align_corners: bool

    def to_mlir(self, inputs, outputs):
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        scales_attr, output_shape = self.get_scales_and_shape(inputs)
        coord_trans = "kALIGN_CORNERS" if self.align_corners else "kHALF_PIXEL"
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get(coord_trans)
        result = tensorrt.resize_linear(
            outputs[0],
            inputs[0],
            coord_trans_attr,
            selector_attr,
            output_shape=output_shape,
            scales=scales_attr,
        )
        return [result]
