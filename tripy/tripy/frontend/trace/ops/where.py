#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tripy import utils
import tripy.frontend.trace.ops.utils as op_utils
from tripy import export
from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Where(BaseTraceOp):

    def infer_shape_output_idxs(self, inputs):
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
            return Result.ok([0])
        elif not isinstance(inputs[1], Shape) and not isinstance(inputs[2], Shape):
            return Result.ok([])
        else:
            return Result.err(
                [
                    "Both value inputs to operator 'where' must either both be tp.Shape or both not be tp.Shape.",
                    f"Given types {type(inputs[1])} and {type(inputs[2])}",
                ]
            )

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        if self.inputs[0].dtype != datatype.bool:
            utils.raise_error_io_info(
                self,
                "Condition input must have boolean type.",
                details=[
                    f"Condition input (input 0) for operation: 'where' must have boolean type, but got: ",
                    self.inputs[0].dtype,
                ],
            )

        op_utils.check_input_dtypes_match(self, op_details="where", start_index=1)
        self.outputs[0].dtype = self.inputs[1].dtype

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import bool as tp_bool, int32
        from tripy.frontend.trace.ops.binary_elementwise import Comparison
        from tripy.flat_ir.ops import CompareOp
        from tripy.flat_ir.ops import SelectOp
        from tripy.flat_ir.ops import MaxOp

        # Unconditionally insert broadcast for all operands
        assert len(inputs) == 3, f"Where op expects 3 inputs but got {len(inputs)}."
        cond_rank, a_rank, b_rank = (input.rank for input in inputs)

        output_rank = max(a_rank, b_rank, cond_rank)
        with FlatIRTensor.context(["make rank of cond, a and b the same."]):
            inputs[0] = op_utils.expand_rank_of_tensor(inputs[0], output_rank - cond_rank)
            inputs[1] = op_utils.expand_rank_of_tensor(inputs[1], output_rank - a_rank)
            inputs[2] = op_utils.expand_rank_of_tensor(inputs[2], output_rank - b_rank)

        with FlatIRTensor.context(["compute element-wise max of input shapes to get the desired output shape."]):
            bcast_cond_and_input = op_utils.compute_shape_of_broadcast(
                op_utils.get_shape_of_tensor(inputs[0]),
                op_utils.get_shape_of_tensor(inputs[1]),
                output_rank,
                shape1_name="the 'condition' tensor",
                shape2_name="the 'input' tensor",
            )
            bcast_input_and_other = op_utils.compute_shape_of_broadcast(
                op_utils.get_shape_of_tensor(inputs[1]),
                op_utils.get_shape_of_tensor(inputs[2]),
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

            inputs[0] = op_utils.insert_broadcast(
                inputs[0],
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details=f"first input of 'where' ('condition')",
            )
            inputs[1] = op_utils.insert_broadcast(
                inputs[1],
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="second input of 'where' ('input')",
            )
            inputs[2] = op_utils.insert_broadcast(
                inputs[2],
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="third input of 'where' ('other')",
            )

        SelectOp.build(inputs, outputs)


@export.public_api(document_under="operations/functions")
def where(condition: "tripy.Tensor", input: "tripy.Tensor", other: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Returns a new tensor of elements selected from either ``input`` or ``other``, depending on ``condition``.

    Args:
        condition: The condition tensor. This must have data type :class:`tripy.bool`.
            Where this is ``True``, elements are selected from ``input``.
            Otherwise, elements are selected from ``other``.
        input: Tensor of values selected at indices where condition is ``True``.
        other: Tensor values selected at indices where condition is ``False``.
            This must have the same datatype as ``input``.

    Returns:
        A new tensor with the broadcasted shape and the same data type as ``input`` and ``other``.

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
def masked_fill(input: "tripy.Tensor", mask: "tripy.Tensor", value: numbers.Number) -> "tripy.Tensor":
    r"""
    Returns a new tensor filled with ``value`` where ``mask`` is ``True`` and elements from
    the input tensor otherwise.

    Args:
        input: The input tensor.
        mask: The mask tensor. This should have data type :class:`tripy.bool`.
        value: the value to fill with. This will be casted to match the data type of the input tensor.

    Returns:
        A new tensor of the same shape and data type as the input tensor.

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
