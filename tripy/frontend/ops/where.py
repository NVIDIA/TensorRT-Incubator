import tripy.frontend.ops.utils as op_utils
from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


class Where(BaseOperator):
    """
    Represents a select operation.
    """

    def infer_shapes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        op_utils.check_input_shapes_match(self, op_details="where")
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        if self.inputs[0].dtype != datatype.bool:
            op_utils.raise_error_io_info(
                self,
                "Condition input must have boolean type.",
                details=[
                    f"Condition input (input 0) for operation: 'where' must have boolean type, but got: ",
                    self.inputs[0].dtype,
                    ".",
                ],
            )

        op_utils.check_input_dtypes_match(self, op_details="where", start_index=1)
        self.outputs[0].dtype = self.inputs[1].dtype

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import SelectOp

        flat_ir.add_op(self, SelectOp, self.inputs, self.outputs)


def where(condition: "tripy.Tensor", x: "tripy.Tensor", y: "tripy.Tensor"):
    """
    Returns a tensor of elements selected from either x or y, depending on condition.

    Args:
        condition: Tensor of bool type, when True, yield x, otherwise yield y
        x: Tensor of values selected at indices where condition is True
        y: Tensor values selected at indices where condition is False

    Returns:
        Output Tensor with selected values.

    Example:
    ::

        import numpy as np

        condition = tp.arange([2, 2], 0) >= tp.arange([2, 2], 1)
        # print(condition)
        # [[True, False],
        #  [True, True]]
        x = tp.ones([2, 2])
        y = tp.zeros([2, 2])
        a = tp.where(condition, x, y)
        assert (a.numpy() == np.array([[1, 0], [1, 1]], dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([condition, x, y], Where)


@TENSOR_METHOD_REGISTRY("masked_fill")
def masked_fill(self: "tripy.Tensor", mask: "tripy.Tensor", value: float) -> "tripy.Tensor":
    """
    Fills elements of tensor with value where mask is True.

    Args:
        mask: Tensor of bool type
        value: the value to fill in with, will be converted to match dtype of self Tensor

    Returns:
        the filled Tensor

    Example:
    ::

        import numpy as np

        mask = tp.arange([2, 2], 0) >= tp.arange([2, 2], 1)
        # print(mask)
        # [[True, False],
        #  [True, True]]
        a = tp.ones([2, 2])
        out = a.masked_fill(mask, -1.0)
        assert (out.numpy() == np.array([[-1, 1], [-1, -1]], dtype=np.float32)).all()
    """
    from tripy.frontend.ops.fill import full_like

    fill_tensor = full_like(self, value)
    return where(mask, fill_tensor, self)
