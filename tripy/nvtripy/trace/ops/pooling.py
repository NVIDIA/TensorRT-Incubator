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


def make_pooling_op(name, pooling_type_name):
    @dataclass(repr=False)
    class PoolingOp(TraceOp):

        kernel_dims: Sequence[int]
        stride: Sequence[int]
        pre_padding: Sequence[Sequence[int]]
        post_padding: Sequence[Sequence[int]]
        avg_excludes_padding: Optional[bool] = None

        infer_rank = op_utils.InferRankPolicies.same_as_input()

        def to_mlir(self, inputs, outputs):
            window_size_attr = ir.DenseI64ArrayAttr.get(self.kernel_dims)
            stride_attr = ir.DenseI64ArrayAttr.get(self.stride)
            pre_padding_attr = ir.DenseI64ArrayAttr.get(self.pre_padding)
            post_padding_attr = ir.DenseI64ArrayAttr.get(self.post_padding)
            pooling_type_attr = tensorrt.PoolingTypeAttr.get(pooling_type_name)
            avg_excludes_padding_attr = None
            if pooling_type_name != "kMAX":
                avg_excludes_padding_attr = ir.BoolAttr.get(self.avg_excludes_padding)

            return [
                tensorrt.pooling(
                    inputs[0],
                    window_size_attr,
                    stride_attr,
                    pre_padding_attr,
                    post_padding_attr,
                    pooling_type_attr,
                    average_count_excludes_padding=avg_excludes_padding_attr,
                )
            ]

    PoolingOp.__name__ = name
    return PoolingOp


MaxPooling = make_pooling_op("MaxPooling", "kMAX")
AvgPooling = make_pooling_op("AvgPooling", "kAVERAGE")
