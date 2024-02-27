import numbers

from tripy import utils
import tripy.frontend.trace.ops.utils as op_utils
from tripy.common import datatype
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


class Where(BaseTraceOp):
    """
    Represents a select operation.
    """

    def infer_shapes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        out_rank = max(len(self.inputs[0].shape), len(self.inputs[1].shape), len(self.inputs[2].shape))
        self.outputs[0].shape = utils.to_dims([-1] * out_rank)

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
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import SelectOp
        from tripy.flat_ir.ops import MaxOp

        # Unconditionally insert broadcast for all operands
        cond_rank, a_rank, b_rank = (len(input.shape) for input in inputs)

        # Make rank of cond, a and b the same.
        output_rank = max(a_rank, b_rank, cond_rank)
        inputs[0] = op_utils.expand_rank_of_tensor(self, inputs[0], output_rank - len(inputs[0].shape))
        inputs[1] = op_utils.expand_rank_of_tensor(self, inputs[1], output_rank - len(inputs[1].shape))
        inputs[2] = op_utils.expand_rank_of_tensor(self, inputs[2], output_rank - len(inputs[2].shape))

        # Compute element-wise max of input shapes to get the desired output shape.
        max_of_cond_and_a_shape = FlatIRTensor.build(shape=inputs[0].shape, dtype=int32, device=inputs[0].device)
        max_of_a_and_b_shape = FlatIRTensor.build(shape=inputs[0].shape, dtype=int32, device=inputs[0].device)
        MaxOp(
            self,
            [op_utils.get_shape_of_tensor(self, inputs[0]), op_utils.get_shape_of_tensor(self, inputs[1])],
            [max_of_cond_and_a_shape],
        )

        MaxOp(
            self,
            [op_utils.get_shape_of_tensor(self, inputs[1]), op_utils.get_shape_of_tensor(self, inputs[2])],
            [max_of_a_and_b_shape],
        )

        inputs[0] = op_utils.insert_broadcast(
            self, inputs[0], outputs[0].shape, use_dynamic_variant=True, shape_of_target_tensor=max_of_a_and_b_shape
        )
        inputs[1] = op_utils.insert_broadcast(
            self, inputs[1], outputs[0].shape, use_dynamic_variant=True, shape_of_target_tensor=max_of_a_and_b_shape
        )
        inputs[2] = op_utils.insert_broadcast(
            self, inputs[2], outputs[0].shape, use_dynamic_variant=True, shape_of_target_tensor=max_of_a_and_b_shape
        )

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
    from tripy.frontend.trace.ops.fill import full_like

    fill_tensor = full_like(self, value)
    return where(mask, fill_tensor, self)
