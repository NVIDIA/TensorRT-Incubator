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
from typing import Optional

from mlir_tensorrt.compiler import ir
from mlir_tensorrt.compiler.dialects import tensorrt
from nvtripy.common import datatype
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import TraceOp


@dataclass(repr=False)
class Attention(TraceOp):
    normalization_operation: Optional[str] = "kSOFTMAX"
    causal: bool = False
    decomposable: bool = False
    normalization_quantize_to_type: Optional[datatype.dtype] = None

    infer_rank = op_utils.InferRankPolicies.same_shape_as_input(0)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_mlir(self, inputs, outputs):
        assert len(inputs) >= 3, "Attention operation should have at least 3 inputs!"

        query, key, value = inputs[0], inputs[1], inputs[2]
        mask = inputs[3] if len(inputs) > 3 else None
        normalization_quantize_scale = inputs[4] if len(inputs) > 4 else None

        # Create attributes
        normalization_operation_attr = None
        if self.normalization_operation:
            norm_op_str = "k" + self.normalization_operation.upper()
            normalization_operation_attr = tensorrt.AttentionNormalizationOpAttr.get(norm_op_str)

        causal_attr = None
        if self.causal:
            causal_attr = ir.BoolAttr.get(self.causal)

        decomposable_attr = None
        if self.decomposable:
            decomposable_attr = ir.BoolAttr.get(self.decomposable)

        normalization_quantize_to_type_attr = None
        if self.normalization_quantize_to_type:
            trt_dtype_str = op_utils.get_trt_dtype_enum_str(self.normalization_quantize_to_type)
            normalization_quantize_to_type_attr = tensorrt.DataTypeAttr.get(trt_dtype_str)

        return [
            tensorrt.attention(
                outputs[0],
                query,
                key,
                value,
                mask=mask,
                normalization_quantize_scale=normalization_quantize_scale,
                normalization_operation=normalization_operation_attr,
                causal=causal_attr,
                decomposable=decomposable_attr,
                normalization_quantize_to_type=normalization_quantize_to_type_attr,
            )
        ]
