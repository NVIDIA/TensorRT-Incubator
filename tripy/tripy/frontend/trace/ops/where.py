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

import tripy.frontend.trace.ops.utils as op_utils
from tripy import constraints, export
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Where(BaseTraceOp):

    def infer_tensor_variants(self, inputs):
        from tripy.frontend.shape import Shape
        from tripy.utils import Result

        # consider the result a shape if (both) the two value arguments are shapes
        if isinstance(inputs[1], Shape) and isinstance(inputs[2], Shape):
            # also require the bool input to be rank 1 to avoid broadcasting to a larger size
            if inputs[0].rank != 1:
                return Result.err(
                    [
                        "If the value inputs to operator 'where' are tp.Shape,"
                        f" the Boolean input must be rank 1, but given rank {inputs[0].rank}",
                    ]
                )
            return Result.ok([Shape])
        elif not isinstance(inputs[1], Shape) and not isinstance(inputs[2], Shape):
            return Result.ok([None])
        else:
            return Result.err(
                [
                    "Both value inputs to operator 'where' must either both be tp.Shape or both not be tp.Shape.",
                    f"Given types {type(inputs[1])} and {type(inputs[2])}",
                ]
            )

    def infer_len(self):
        # broadcast to the largest of the input lengths
        return [max(map(lambda inp: op_utils.get_trace_shape(inp)[0], self.inputs))]

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        self.outputs[0].dtype = self.inputs[1].dtype

    @frontend_utils.make_function
    def to_flat_ir(self, inputs, outputs):
        from tripy.common.datatype import bool as tp_bool
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import CompareOp, MaxOp, SelectOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.frontend.trace.ops.binary_elementwise import Comparison

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
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"condition": "T2", "input": "T1", "other": "T1", constraints.RETURN_VALUE: "T1"},
)
def where(condition: "tripy.Tensor", input: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
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
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
    dtype_constraints={"input": "T1", "mask": "T2", constraints.RETURN_VALUE: "T1"},
)
def masked_fill(input: "tripy.Tensor", mask: "tripy.Tensor", value: numbers.Number) -> "tripy.Tensor":
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
    from tripy.frontend.trace.ops.fill import full_like

    fill_tensor = full_like(input, value)
    return where(mask, fill_tensor, input)
