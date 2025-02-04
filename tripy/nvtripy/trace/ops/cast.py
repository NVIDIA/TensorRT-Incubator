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

from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Cast(BaseTraceOp):
    dtype: "nvtripy.common.dtype"

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def to_flat_ir(self, inputs, outputs):
        import nvtripy.trace.ops.utils as op_utils
        from nvtripy.common.datatype import bool as tp_bool
        from nvtripy.common.datatype import float32, int32, int64
        from nvtripy.flat_ir.ops import CompareOp, ConstantOp, ConvertOp, DynamicBroadcastOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # If we need to create a constant (namely for comparing with zero), it has to use one of these dtypes.
        # If the input is not one of these dtypes, the constant needs to be created in one of these and converted.
        DTYPES_FOR_CONSTANTS = {float32, int32, int64}

        convert_input = inputs[0]

        # For conversion to bool, we must compare with 0 since the underlying semantics for StableHLO
        # are to do truncation for conversion to integer types (and bools are i1). This would get
        # unintended results for even numbers, which truncate to 0 in i1.
        if self.dtype == tp_bool:
            # Creating a zero tensor uses the same logic as the zeros_like initializer

            # If the input dtype does not allow directly creating a Tripy array, we have to use another like f32
            # and then cast the zeros tensor.
            zero_dtype = convert_input.dtype if convert_input.dtype in DTYPES_FOR_CONSTANTS else float32
            single_zero = FlatIRTensor.build(
                shape=[],
                rank=0,
                dtype=zero_dtype,
                device=convert_input.device,
                reason_details=["Zero scalar for casting to bool"],
            )
            ConstantOp.build([], [single_zero], data=0)
            zeros_shape = op_utils.get_shape_of_tensor(convert_input)
            zeros = FlatIRTensor.build(
                shape=convert_input.shape,
                rank=convert_input.rank,
                dtype=zero_dtype,
                device=convert_input.device,
                reason_details=["Tensor of zeroes for comparing to cast to bool"],
            )
            DynamicBroadcastOp.build([single_zero, zeros_shape], [zeros], broadcast_dim=[])

            if zero_dtype != convert_input.dtype:
                zero_output = FlatIRTensor.build(
                    shape=zeros.shape,
                    rank=zeros.rank,
                    dtype=convert_input.dtype,
                    device=zeros.device,
                    reason_details=[
                        f"Cast zero tensor because it cannot be created directly from array with dtype {convert_input.dtype}"
                    ],
                )
                ConvertOp.build([zeros], [zero_output])
                zeros = zero_output

            CompareOp.build([convert_input, zeros], outputs, compare_direction="NE")
            return

        ConvertOp.build([convert_input], outputs)
