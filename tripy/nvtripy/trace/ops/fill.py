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
class Fill(BaseTraceOp):
    dtype: datatype.dtype

    infer_rank = op_utils.InferRankPolicies.same_as_shape_of_shape_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from nvtripy.common import device

        self.outputs[0].device = device.fast_init("gpu", 0)

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import ConvertOp, DynamicBroadcastOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        const_val_tensor = None
        assert (
            len(inputs) == 2
        ), f"Expected value of Fill to be provided as input. Expected 2 inputs, got {len(inputs)}."
        const_val_tensor = inputs[1]
        if inputs[1].dtype != outputs[0].dtype:
            out = FlatIRTensor.build(
                shape=(),
                rank=0,
                dtype=outputs[0].dtype,
                device=outputs[0].device,
                reason_details=[f"create the constant value tensor for a fill operation"],
            )

            ConvertOp.build([const_val_tensor], [out])
            const_val_tensor = out
        DynamicBroadcastOp.build(
            [const_val_tensor, inputs[0]],
            outputs,
            broadcast_dim=[],
        )
