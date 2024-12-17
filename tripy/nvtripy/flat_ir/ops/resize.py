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

from typing import Sequence
from dataclasses import dataclass

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import tensorrt

from nvtripy.flat_ir.ops.base import BaseFlatIROp


@dataclass(repr=False)
class ResizeNearestOp(BaseFlatIROp):

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get("kASYMMETRIC")
        rounding_mode_attr = tensorrt.ResizeRoundModeAttr.get("kFLOOR")
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        return [
            tensorrt.resize_nearest(
                out_type,
                operands[0],
                coord_trans_attr,
                rounding_mode_attr,
                selector_attr,
                output_shape=operands[1],
            )
        ]


@dataclass(repr=False)
class ResizeLinearOp(BaseFlatIROp):

    align_corners: bool

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        coord_trans = "kALIGN_CORNERS" if self.align_corners else "kHALF_PIXEL"
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get(coord_trans)
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        return [
            tensorrt.resize_linear(
                out_type,
                operands[0],
                coord_trans_attr,
                selector_attr,
                output_shape=operands[1],
            )
        ]


@dataclass(repr=False)
class ResizeCubicOp(BaseFlatIROp):

    align_corners: bool
    cubic_coeff: float

    def to_mlir(self, operands):
        out_type = self.outputs[0].to_mlir()
        coord_trans = "kALIGN_CORNERS" if self.align_corners else "kHALF_PIXEL"
        coord_trans_attr = tensorrt.ResizeCoordinateTransformationAttr.get(coord_trans)
        selector_attr = tensorrt.ResizeSelectorAttr.get("kFORMULA")
        cubic_coeff_attr = ir.FloatAttr.get(ir.F32Type.get(), self.cubic_coeff)
        return [
            tensorrt.resize_cubic(
                out_type,
                operands[0],
                coord_trans_attr,
                selector_attr,
                cubic_coeff_attr,
                output_shape=operands[1],
            )
        ]
