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
class Where(BaseTraceOp):
    infer_rank = op_utils.InferRankPolicies.max_of_inputs()

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        self.outputs[0].dtype = self.inputs[1].dtype

    def to_flat_ir(self, inputs, outputs):
        from nvtripy.flat_ir.ops import SelectOp
        from nvtripy.flat_ir.tensor import FlatIRTensor

        # Unconditionally insert broadcast for all operands
        assert len(inputs) == 3, f"Where op expects 3 inputs but got {len(inputs)}."
        cond_rank, a_rank, b_rank = (input.rank for input in inputs)

        output_rank = max(a_rank, b_rank, cond_rank)
        with FlatIRTensor.context(["make rank of cond, a and b the same."]):
            broadcasted_input_0 = op_utils.expand_rank_of_tensor(inputs[0], output_rank - cond_rank)
            broadcasted_input_1 = op_utils.expand_rank_of_tensor(inputs[1], output_rank - a_rank)
            broadcasted_input_2 = op_utils.expand_rank_of_tensor(inputs[2], output_rank - b_rank)

        with FlatIRTensor.context(["compute element-wise max of input shapes to get the desired output shape."]):
            bcast_cond_and_input = op_utils.compute_shape_of_broadcast(
                op_utils.get_shape_of_tensor(broadcasted_input_0),
                op_utils.get_shape_of_tensor(broadcasted_input_1),
                output_rank,
                shape1_name="the 'condition' tensor",
                shape2_name="the 'input' tensor",
            )
            bcast_input_and_other = op_utils.compute_shape_of_broadcast(
                op_utils.get_shape_of_tensor(broadcasted_input_1),
                op_utils.get_shape_of_tensor(broadcasted_input_2),
                output_rank,
                shape1_name="the 'input' tensor",
                shape2_name="the 'other' tensor",
            )
            computed_output_shape = op_utils.compute_shape_of_broadcast(
                bcast_cond_and_input,
                bcast_input_and_other,
                output_rank,
                shape1_name="the previously computed broadcast of the 'condition' and 'input' tensor",
                shape2_name="the previously computed broadcast of the 'input' and 'other' tensors",
            )

            broadcasted_input_0 = op_utils.insert_broadcast(
                broadcasted_input_0,
                outputs[0].rank,
                shape_of_target_tensor=computed_output_shape,
                tensor_details=f"first input of 'where' ('condition')",
            )
            broadcasted_input_1 = op_utils.insert_broadcast(
                broadcasted_input_1,
                outputs[0].rank,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="second input of 'where' ('input')",
            )
            broadcasted_input_2 = op_utils.insert_broadcast(
                broadcasted_input_2,
                outputs[0].rank,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="third input of 'where' ('other')",
            )

        SelectOp.build([broadcasted_input_0, broadcasted_input_1, broadcasted_input_2], outputs)
