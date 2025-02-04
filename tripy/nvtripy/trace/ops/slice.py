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
class Slice(BaseTraceOp):
    infer_rank = op_utils.InferRankPolicies.same_as_input()

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import bool as tp_bool
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import DynamicReshapeOp, DynamicSliceOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        with FlatIRTensor.context(["construct constant tensors for slice `dim`'s > len(slice_params) // 3"]):
            device = inputs[0].device
            zero_1d = op_utils.add_constant_tensor_from_list([0], device)
            one_1d = op_utils.add_constant_tensor_from_list([1], device)

        data_tensor = inputs[0]
        slice_params = inputs[1:]
        input_rank = data_tensor.rank
        input_shape = op_utils.get_shape_of_tensor(data_tensor)

        start_idxs = []
        limit_idxs = []
        stride_idxs = []

        for dim in range(input_rank):
            with FlatIRTensor.context([f"generate slice index tensors for dimension {dim}"]):
                shape_slice = op_utils.slice_rank1_tensor(
                    input_shape,
                    dim,
                    reason_details=[
                        "slicing the shape tensor ",
                        input_shape,
                        f" to get the dimension with index {dim}",
                    ],
                )

                if dim < len(slice_params) // 3:

                    def expand_to_rank1(index_tensor):
                        reshape_out = FlatIRTensor.build(
                            shape=[1],
                            rank=1,
                            dtype=int32,
                            device=device,
                            reason_details=["reshape index tensor into singleton in case it is () instead of (1,)"],
                        )
                        shape_input = op_utils.add_constant_tensor_from_list([1], device)
                        DynamicReshapeOp.build([index_tensor, shape_input], [reshape_out])
                        return reshape_out

                    # if start > limit, the dim should be empty (we will set start to match the end)
                    def adjust_start(start_bound, end_bound):
                        from nvtripy.flat_ir.ops import CompareOp, SelectOp
                        from nvtripy.trace.ops.binary_elementwise import Comparison

                        start_comparison = FlatIRTensor.build(
                            shape=[1],
                            rank=1,
                            dtype=tp_bool,
                            device=device,
                            reason_details=["Check if start > end"],
                        )
                        adjusted_start = FlatIRTensor.build(
                            shape=[1],
                            rank=1,
                            dtype=int32,
                            device=device,
                            reason_details=["Shift the start to the end so we get an empty dimension if start > end"],
                        )

                        # pick start if it is <= end
                        CompareOp.build(
                            [start_bound, end_bound],
                            [start_comparison],
                            compare_direction=Comparison.Kind.LESS_EQUAL.compare_direction,
                        )
                        SelectOp.build([start_comparison, start_bound, end_bound], [adjusted_start])
                        return adjusted_start

                    start_bound = expand_to_rank1(slice_params[3 * dim])
                    end_bound = expand_to_rank1(slice_params[3 * dim + 1])

                    start_idxs.append(adjust_start(start_bound, end_bound))
                    limit_idxs.append(end_bound)
                    stride_idxs.append(expand_to_rank1(slice_params[3 * dim + 2]))
                else:
                    start_idxs.append(zero_1d)
                    limit_idxs.append(shape_slice)
                    stride_idxs.append(one_1d)

        with FlatIRTensor.context(["concatenate slice index tensors"]):
            start_index_tensor = op_utils.concatenate_tensors(start_idxs, dim=0)
            limit_index_tensor = op_utils.concatenate_tensors(limit_idxs, dim=0)
            stride_index_tensor = op_utils.concatenate_tensors(stride_idxs, dim=0)

        DynamicSliceOp.build([data_tensor, start_index_tensor, limit_index_tensor, stride_index_tensor], outputs)
