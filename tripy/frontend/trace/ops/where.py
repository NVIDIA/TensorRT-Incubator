import numbers
from dataclasses import dataclass

from tripy import utils
import tripy.frontend.trace.ops.utils as op_utils
from tripy import export
from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Where(BaseTraceOp):

    def get_operand_shape_after_broadcast(self, cond_shape, a_shape, b_shape):
        def broadcast_equivalent_shape(a, b):
            shapes = op_utils.get_broadcast_compatible_shapes(a, b)
            bcast_check = op_utils.is_broadcast_compatible(*shapes)
            if not bcast_check:
                op_utils.raise_error_io_info(
                    self,
                    "Input tensors are not broadcast compatible.",
                    details=[
                        "Input tensors for where operation must be broadcast compatible but ",
                    ]
                    + bcast_check.error_details,
                )
            return tuple(op_utils.get_broadcast_dim(*d) for d in zip(*shapes))

        cond_shape = broadcast_equivalent_shape(cond_shape, a_shape)
        cond_shape = broadcast_equivalent_shape(cond_shape, b_shape)

        return cond_shape

    def infer_shapes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"

        # Output shape is broadcast of all 3 input tensor shapes.
        operand_shape = self.get_operand_shape_after_broadcast(*[inp.shape for inp in self.inputs])
        self.outputs[0].shape = operand_shape

        # TODO: https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/issues/152 will remove get_operand_shape_after_broadcast and line 38-39 and replace with line 42-43.
        # out_rank = max(len(self.inputs[0].shape), len(self.inputs[1].shape), len(self.inputs[2].shape))
        # self.outputs[0].shape = utils.to_dims([-1] * out_rank)

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        if self.inputs[0].dtype != datatype.bool:
            op_utils.raise_error_io_info(
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
            inputs[0] = op_utils.expand_rank_of_tensor(inputs[0], output_rank - len(inputs[0].shape))
            inputs[1] = op_utils.expand_rank_of_tensor(inputs[1], output_rank - len(inputs[1].shape))
            inputs[2] = op_utils.expand_rank_of_tensor(inputs[2], output_rank - len(inputs[2].shape))

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
                outputs[0].shape,
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details=f"first input of 'where' ('condition')",
            )
            inputs[1] = op_utils.insert_broadcast(
                inputs[1],
                outputs[0].shape,
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="second input of 'where' ('input')",
            )
            inputs[2] = op_utils.insert_broadcast(
                inputs[2],
                outputs[0].shape,
                outputs[0].rank,
                use_dynamic_variant=True,
                shape_of_target_tensor=computed_output_shape,
                tensor_details="third input of 'where' ('other')",
            )

        SelectOp.build(inputs, outputs)


@export.public_api(document_under="tensor_operations")
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


@export.public_api(document_under="tensor_operations")
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
