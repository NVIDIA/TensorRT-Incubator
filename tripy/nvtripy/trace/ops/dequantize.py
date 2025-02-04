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

from nvtripy.common import datatype
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Dequantize(BaseTraceOp):

    dtype: datatype.dtype
    dim: int

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import ConcatenateOp, ConvertOp, DivideOp, DynamicBroadcastOp, DynamicReshapeOp, MulOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # Represent quantize as convert(input, dtype) * scale
        converted_tensor = FlatIRTensor.build(
            shape=inputs[0].shape,
            rank=inputs[0].rank,
            dtype=self.dtype,
            device=inputs[0].device,
            reason_details=["Convert the input tensor to dequantized dtype."],
        )
        ConvertOp.build([inputs[0]], [converted_tensor])

        broadcast_scale = FlatIRTensor.build(
            shape=inputs[0].shape,  # broadcast to input's shape
            rank=inputs[0].rank,
            dtype=inputs[1].dtype,  # original scale's dtype
            device=inputs[1].device,
            reason_details=["Broadcast the scale to the input's shape in dequant operation."],
        )
        if inputs[1].rank == 0 or inputs[1].rank == 1:
            shape_of_input = op_utils.get_shape_of_tensor(inputs[0])
            broadcast_dim = [self.dim] if self.dim is not None else []
            DynamicBroadcastOp.build([inputs[1], shape_of_input], [broadcast_scale], broadcast_dim=broadcast_dim)
        else:
            # block-wise quant, input: [block_size * A, B], scale: [A, B]
            # Broadcast(scale) -> [block_size, A, B]
            # Reshape(scale) -> [block_size * A, B]
            # Mul(input, scale)
            num_blocks = FlatIRTensor.build(
                shape=(1,),
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["Compute the number of blocks in block-wise dequantization"],
            )
            blocked_shape = FlatIRTensor.build(
                shape=(3,),
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["Compute shape with an extra blocked_size dimension."],
            )
            blocked_scale = FlatIRTensor.build(
                rank=3,
                dtype=inputs[1].dtype,
                device=inputs[1].device,
                reason_details=["Construct the scale to have an extra block_size dimension."],
            )

            input_dim0 = op_utils.get_dim_size_1d_tensor(inputs[0], dim=0)
            scale_dim0 = op_utils.get_dim_size_1d_tensor(inputs[1], dim=0)
            feat_dim = op_utils.get_dim_size_1d_tensor(inputs[1], dim=1)
            DivideOp.build([input_dim0, scale_dim0], [num_blocks])
            ConcatenateOp.build([num_blocks, scale_dim0, feat_dim], [blocked_shape], dim=0)
            DynamicBroadcastOp.build([inputs[1], blocked_shape], [blocked_scale], broadcast_dim=[1, 2])
            origin_input_shape = op_utils.get_shape_of_tensor(inputs[0])
            DynamicReshapeOp.build([blocked_scale, origin_input_shape], [broadcast_scale])

        MulOp.build([converted_tensor, broadcast_scale], outputs)
