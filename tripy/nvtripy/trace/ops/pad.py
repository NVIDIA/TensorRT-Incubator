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
from typing import Union

from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Pad(BaseTraceOp):

    padding_value: Union[int, float]

    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import ConstantOp, DynamicPadOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        pad_val_tensor = FlatIRTensor.build(
            shape=(),
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor (containing {self.padding_value}) for a pad operation"],
        )
        ConstantOp.build([], [pad_val_tensor], data=self.padding_value)

        # interior_padding is not supported
        # create the default value
        pad_size_shape = (inputs[0].rank,)
        interior_pad_tensor = FlatIRTensor.build(
            shape=pad_size_shape,
            rank=1,
            dtype=int32,
            device=outputs[0].device,
            reason_details=[f"create the default value for interior_padding argument."],
        )
        ConstantOp.build([], [interior_pad_tensor], data=[0] * inputs[0].rank)

        # [operand, pad_val, low, high, interior]
        inputs.insert(1, pad_val_tensor)
        inputs.append(interior_pad_tensor)
        # set padding size tensors' shape
        # because stablehlo requires static shapes
        inputs[2].shape = pad_size_shape
        inputs[3].shape = pad_size_shape
        DynamicPadOp.build(inputs, outputs)
