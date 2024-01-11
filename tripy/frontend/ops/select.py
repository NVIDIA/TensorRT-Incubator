from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY


class Select(BaseOperator):
    """
    Represents a select operation.
    """

    def to_trace_str(self):
        return f"{self.outputs[0].name} = Tensor.select(condition={self.inputs[0].name}, x={self.inputs[1].name}, y={self.inputs[2].name})"

    def infer_shapes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        assert (
            self.inputs[0].shape == self.inputs[1].shape and self.inputs[0].shape == self.inputs[2].shape
        ), f"Input shapes for Select do not match: condition={self.inputs[0].shape}, x={self.inputs[1].shape}, y={self.inputs[2].shape}"
        self.outputs[0].shape = self.inputs[0].shape

    def infer_dtypes(self):
        assert len(self.inputs) == 3, "Select operation should have exactly 3 inputs!"
        if self.inputs[0].dtype != datatype.bool:
            raise TypeError(f"Condition of Select must be bool type, got {self.inputs[0].dtype}")
        if self.inputs[1].dtype != self.inputs[2].dtype:
            raise TypeError(f"Select's input datatypes mismatch, got {self.inputs[1].dtype} and {self.inputs[2].dtype}")
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
        # print(condition.eval().view())
        # [[True, False],
        #  [True, True]]
        x = tp.ones([2, 2])
        y = tp.zeros([2, 2])
        a = tp.where(condition, x, y)
        assert (a.numpy() == np.array([[1, 0], [1, 1]], dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([condition, x, y], Select)


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
        # print(mask.eval().view())
        # [[True, False],
        #  [True, True]]
        a = tp.ones([2, 2])
        out = a.masked_fill(mask, -1.0)
        assert (out.numpy() == np.array([[-1, 1], [-1, -1]], dtype=np.float32)).all()
    """
    from tripy.frontend.ops.fill import full_like

    return where(mask, full_like(self, value), self)
