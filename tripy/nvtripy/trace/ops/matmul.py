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
from typing import Dict, List

from nvtripy.trace.ops import utils as op_utils
from nvtripy.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class MatrixMultiplication(BaseTraceOp):

    contracting_dim: Dict[str, List[int]]
    batching_dim: Dict[str, List[int]]

    def __str__(self):
        return f"{self.outputs[0].name} = {' @ '.join([inp.name for inp in self.inputs])}"

    def infer_rank(self):
        out_rank = max(self.inputs[0].rank, self.inputs[1].rank)
        if self.inputs[0].rank == 1 and self.inputs[1].rank == 1:
            out_rank = 0
        elif self.inputs[0].rank == 1 or self.inputs[1].rank == 1:
            out_rank = 1
        self.outputs[0].rank = out_rank

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.common.datatype import int32
        from nvtripy.flat_ir.ops import ConcatenateOp, DotOp, DynamicSliceOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # Following steps are followed in the implementation below:
        # 1. Slice the input shape into batch dims and matrix dims.
        # 2. Ensure that batch dims for both operands are of same rank by prepending shape tensor with 1s.
        # 3. Apply broadcasting rule to batch dims to get the output batch dims.
        # 4. Concatenate the batch dims with matrix dims computed in step1.
        # 5. Use the computed output dims from #4 to broadcast both the inputs.
        # 6. Invocate DotOp.

        def split_shape_in_batch_and_mat_dims(input: "FlatIRTensor", nb_batch_dims: int):
            input_shape = op_utils.get_shape_of_tensor(input)

            zero_1d = op_utils.add_constant_tensor_from_list([0], input.device)
            one_1d = op_utils.add_constant_tensor_from_list([1], input.device)
            slice_len = op_utils.add_constant_tensor_from_list([nb_batch_dims], input.device)
            batch_slice = FlatIRTensor.build(
                dtype=int32,
                device=input.device,
                rank=1,
                shape=[nb_batch_dims],
                reason_details=["slice the input shape ", input_shape, " to get batch dims."],
            )
            DynamicSliceOp.build([input_shape, zero_1d, slice_len, one_1d], [batch_slice])

            end_len = op_utils.add_constant_tensor_from_list([input.rank], input.device)
            mat_slice = FlatIRTensor.build(
                rank=1,
                dtype=int32,
                device=input.device,
                reason_details=["slice the input shape ", input_shape, " into mat dims."],
            )
            DynamicSliceOp.build([input_shape, slice_len, end_len, one_1d], [mat_slice])
            return batch_slice, mat_slice

        def append_ones_data_tensor(input, nb_ones):
            extra_a_ones = op_utils.add_constant_tensor_from_list([1] * nb_ones, input.device)
            input_expanded = FlatIRTensor.build(
                rank=1,
                dtype=int32,
                device=input.device,
                reason_details=[f"append {nb_ones} ones to the shape tensor ", input],
            )
            ConcatenateOp.build([extra_a_ones, input], [input_expanded], dim=0)
            return input_expanded

        broadcasted_inputs_0 = inputs[0]
        broadcasted_inputs_1 = inputs[1]

        a_rank, b_rank = (input.rank for input in inputs)
        if not (a_rank == 2 and b_rank == 2):
            nb_a_batch_dims, nb_b_batch_dims = [max(rank - 2, 0) for rank in [a_rank, b_rank]]
            nb_result_batch_dims = max(nb_a_batch_dims, nb_b_batch_dims)

            with FlatIRTensor.context(["split the input shapes into batch dims and matrix dims"]):
                a_batch_shape, a_mat_shape = split_shape_in_batch_and_mat_dims(inputs[0], nb_a_batch_dims)
                b_batch_shape, b_mat_shape = split_shape_in_batch_and_mat_dims(inputs[1], nb_b_batch_dims)

            with FlatIRTensor.context(
                ["Ensure that batch dims for both operands are of same rank by prepending shape tensor with 1s."]
            ):
                a_batch_shapes_with_ones = append_ones_data_tensor(
                    a_batch_shape, nb_result_batch_dims - nb_a_batch_dims
                )
                b_batch_shapes_with_ones = append_ones_data_tensor(
                    b_batch_shape, nb_result_batch_dims - nb_b_batch_dims
                )

            # Apply broadcasting rule of batch shapes to get the resulting batch shapes
            bcast_of_batch_shapes = op_utils.compute_shape_of_broadcast(
                a_batch_shapes_with_ones,
                b_batch_shapes_with_ones,
                nb_result_batch_dims,
                shape1_name="the batch dims of a",
                shape2_name="the batch dims of b",
            )

            with FlatIRTensor.context(["concatenate batch dims with matrix dims computed in step 1"]):
                a_dims = op_utils.concatenate_tensors([bcast_of_batch_shapes, a_mat_shape], dim=0)
                b_dims = op_utils.concatenate_tensors([bcast_of_batch_shapes, b_mat_shape], dim=0)

            with FlatIRTensor.context(["Use the computed output dims from #4 to broadcast both the inputs."]):
                broadcasted_inputs_0 = op_utils.insert_broadcast(
                    inputs[0],
                    out_rank=nb_result_batch_dims + a_rank - nb_a_batch_dims,
                    shape_of_target_tensor=a_dims,
                    tensor_details=["left operand of DotOp"],
                )
                broadcasted_inputs_1 = op_utils.insert_broadcast(
                    inputs[1],
                    out_rank=nb_result_batch_dims + b_rank - nb_b_batch_dims,
                    shape_of_target_tensor=b_dims,
                    tensor_details=["right operand of DotOp"],
                )

        DotOp.build(
            [broadcasted_inputs_0, broadcasted_inputs_1],
            outputs,
            contracting_dim=self.contracting_dim,
            batching_dim=self.batching_dim,
        )
