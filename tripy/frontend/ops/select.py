from tripy.common import datatype
from tripy.frontend.ops.base import BaseOperator


class Select(BaseOperator):
    """
    Represents a select operation.
    """

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 3, "Select operation should have exactly 3 inputs!"
        assert len(output_names) == 1, "Select operation should have exactly one output!"
        return f"{output_names[0]} = Tensor.select(condition={input_names[0]}, x={input_names[1]}, y={input_names[2]})"

    def infer_shapes(self, input_shapes):
        assert len(input_shapes) == 3, "Select operation should have exactly 3 inputs!"
        assert (
            input_shapes[0] == input_shapes[1] and input_shapes[0] == input_shapes[2]
        ), f"Input shapes for Select do not match: condition={input_shapes[0]}, x={input_shapes[1]}, y={input_shapes[2]}"
        return [input_shapes[0]]

    def infer_dtypes(self, input_dtypes):
        assert len(input_dtypes) == 3, "Select operation should have exactly 3 inputs!"
        if input_dtypes[0] != datatype.bool:
            raise TypeError(f"Condition of Select must be bool type, got {input_dtypes[0]}")
        if input_dtypes[1] != input_dtypes[2]:
            raise TypeError(f"Select's input datatypes mismatch, got {input_dtypes[1]} and {input_dtypes[2]}")
        return [input_dtypes[1]]

    def infer_devices(self, input_devices):
        assert len(input_devices) == 3, "Select operation should have exactly 3 inputs!"
        assert (
            input_devices[0] == input_devices[1] and input_devices[0] == input_devices[2]
        ), f"Input devices for Select do not match: condition={input_devices[0]}, x={input_devices[1]}, y={input_devices[2]}"
        return [input_devices[0]]

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
