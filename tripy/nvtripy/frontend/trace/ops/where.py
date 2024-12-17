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

import numbers
from dataclasses import dataclass

import nvtripy.frontend.trace.ops.utils as op_utils
from nvtripy import export, wrappers
from nvtripy.frontend.trace.ops.base import BaseTraceOp


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


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"condition": "T2", "input": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
)
def where(condition: "nvtripy.Tensor", input: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Returns a new tensor of elements selected from either ``input`` or ``other``, depending on ``condition``.

    Args:
        condition: The condition tensor.
            Where this is ``True``, elements are selected from ``input``.
            Otherwise, elements are selected from ``other``.
        input: Tensor of values selected at indices where condition is ``True``.
        other: Tensor values selected at indices where condition is ``False``.

    Returns:
        A new tensor with the broadcasted shape.

    Constraints:
        All three parameters must be broadcast-compatible with each other.

    .. code-block:: python
        :linenos:
        :caption: Example

        condition = tp.Tensor([[True, False], [True, True]])
        input = tp.ones([2, 2], dtype=tp.float32)
        other = tp.zeros([2, 2], dtype=tp.float32)
        output = tp.where(condition, input, other)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[1, 0], [1, 1]], dtype=np.float32))
    """
    return Where.build([condition, input, other])


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "mask": "T2", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
)
def masked_fill(input: "nvtripy.Tensor", mask: "nvtripy.Tensor", value: numbers.Number) -> "nvtripy.Tensor":
    r"""
    Returns a new tensor filled with ``value`` where ``mask`` is ``True`` and elements from
    the input tensor otherwise.

    Args:
        input: The input tensor.
        mask: The mask tensor.
        value: the value to fill with. This will be casted to match the data type of the input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        mask = tp.Tensor([[True, False], [True, True]])
        input = tp.zeros([2, 2])
        output = tp.masked_fill(input, mask, -1.0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[-1, 0], [-1, -1]], dtype=np.float32))
    """
    from nvtripy.frontend.trace.ops.fill import full_like

    fill_tensor = full_like(input, value)
    return where(mask, fill_tensor, input)
