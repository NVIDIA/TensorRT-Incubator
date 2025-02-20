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

from nvtripy import utils
from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Gather(BaseTraceOp):
    axis: int

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank + self.inputs[1].rank - 1

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_devices(self):
        self.outputs[0].device = self.inputs[0].device

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import DynamicGatherOp, DynamicSliceOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        input_shape = op_utils.get_shape_of_tensor(inputs[0])
        zero_1d = op_utils.add_constant_tensor_from_list([0], inputs[0].device)
        one_1d = op_utils.add_constant_tensor_from_list([1], inputs[0].device)
        second_half_start = op_utils.add_constant_tensor_from_list([self.axis + 1], inputs[0].device)

        # Gather slice_sizes is the same as size of input with dim at self.axis replaced with 1.
        # Code below performs input_shape[0:self.axis], 1, input_shape[self.axis + 1 : ]
        size_partial_tensors = []
        if self.axis > 0:
            slice_limit = op_utils.add_constant_tensor_from_list([self.axis], inputs[0].device)
            axis_first_half = FlatIRTensor.build(
                rank=1,
                shape=[self.axis],
                dtype=int32,
                device=inputs[0].device,
                reason_details=["slice the input shape ", input_shape, " to get input_shape[0:self.axis]."],
            )
            DynamicSliceOp.build([input_shape, zero_1d, slice_limit, one_1d], [axis_first_half])
            size_partial_tensors.append(axis_first_half)

        size_partial_tensors.append(one_1d)
        if self.axis + 1 < inputs[0].rank:
            slice_limit = op_utils.add_constant_tensor_from_list([inputs[0].rank], inputs[0].device)
            axis_second_half = FlatIRTensor.build(
                rank=1,
                dtype=int32,
                device=inputs[0].device,
                reason_details=["slice the input shape ", input_shape, " to get input_shape[self.axis + 1 :]."],
            )
            DynamicSliceOp.build([input_shape, second_half_start, slice_limit, one_1d], [axis_second_half])
            size_partial_tensors.append(axis_second_half)

        slice_sizes = op_utils.concatenate_tensors(size_partial_tensors, dim=0)
        DynamicGatherOp.build([inputs[0], inputs[1], slice_sizes], outputs, self.axis)
