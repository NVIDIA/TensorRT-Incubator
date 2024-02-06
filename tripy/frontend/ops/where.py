import numbers

import tripy.frontend.ops.utils as op_utils
from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


class Where(BaseOperator):
    """
    Represents a select operation.
    """

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
                    + bcast_check.details,
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
        from tripy.flat_ir.ops import SelectOp

        # Unconditionally insert broadcast for all operands
        inputs[0] = op_utils.insert_broadcast(self, inputs[0], outputs[0].shape)
        inputs[1] = op_utils.insert_broadcast(self, inputs[1], outputs[0].shape)
        inputs[2] = op_utils.insert_broadcast(self, inputs[2], outputs[0].shape)

        SelectOp(self, inputs, outputs)


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

        # TODO: Initialize directly from booleans
        condition = tp.iota([2, 2], 0) >= tp.iota([2, 2], 1)

        input = tp.ones([2, 2], dtype=tp.float32)
        other = tp.zeros([2, 2], dtype=tp.float32)
        output = tp.where(condition, input, other)

        assert np.array_equal(output.numpy(), np.array([[1, 0], [1, 1]], dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([condition, input, other], Where)


@TENSOR_METHOD_REGISTRY("masked_fill")
def masked_fill(self, mask: "tripy.Tensor", value: numbers.Number) -> "tripy.Tensor":
    r"""
    Returns a new tensor filled with ``value`` where ``mask`` is ``True`` and elements from
    this tensor otherwise.

    Args:
        mask: The mask tensor. This should have data type :class:`tripy.bool`.
        value: the value to fill with. This will be casted to match the data type of this tensor.

    Returns:
        A new tensor of the same shape and data type as this one.

    .. code-block:: python
        :linenos:
        :caption: Example

        # TODO: Initialize directly from booleans
        mask = tp.iota([2, 2], 0) >= tp.iota([2, 2], 1)

        input = tp.zeros([2, 2])
        output = input.masked_fill(mask, -1.0)

        assert np.array_equal(output.numpy(), np.array([[-1, 0], [-1, -1]], dtype=np.float32))
    """
    from tripy.frontend.ops.fill import full_like

    fill_tensor = full_like(self, value)
    return where(mask, fill_tensor, self)
