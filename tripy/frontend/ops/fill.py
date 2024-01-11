from dataclasses import dataclass

from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.utils import to_dims


@dataclass
class Fill(BaseOperator):
    """
    Represents a fill operation.
    """

    value: float
    shape: ShapeInfo
    dtype: datatype.dtype

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 0 or len(input_names) == 1, "Fill operation should have 0 ir 1 input!"
        assert len(output_names) == 1, "Fill operation should exactly ones output!"
        if len(input_names) == 1:
            return f"{output_names[0]} = Tensor.fill_like(value={self.value}, like={input_names[0]})"
        else:
            return f"{output_names[0]} = Tensor.fill(value={self.value}, shape={self.shape}, dtype={self.dtype.name})"

    def infer_shapes(self):
        if len(self.inputs) == 1:
            self.shape = self.inputs[0].shape
        self.outputs[0].shape = self.shape

    def infer_dtypes(self):
        if len(self.inputs) == 1 and self.dtype is None:
            self.dtype = self.inputs[0].dtype
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, flat_ir):
        import numpy as np
        from tripy.flat_ir.ops import BroadcastOp, ConstantOp

        const_val_tensor = flat_ir.add_tensor(shape=[], dtype=self.outputs[0].dtype, device=self.outputs[0].device)
        flat_ir.add_op(self, ConstantOp, [], [const_val_tensor], data=np.array(self.value, dtype=self.dtype.name))
        flat_ir.add_op(self, BroadcastOp, [const_val_tensor], self.outputs, broadcast_dim=[])


def full(shape: ShapeInfo, fill_value, dtype: datatype.dtype = datatype.float32):
    """
    Creates a Tensor of `shape` filled with `fill_value`.

    Args:
        shape: A list or tuple of integers
        fill_value: A numeric scalar value to fill the resulting Tensor.
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to fill_value.

    Example:
    ::

        import numpy as np

        a = tp.full([2, 3], 2)
        assert (a.numpy() == np.full([2, 3], 2, dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([], Fill, fill_value, to_dims(shape), dtype)


def full_like(input: "tripy.Tensor", fill_value, dtype: datatype.dtype = None):
    """
    Creates a Tensor filled with `fill_value`, its shape (and dtype if not given) are determined by the `input` Tensor.

    Args:
        input: Input Tensor to determine the resulting Tensor's shape (and dtype if not given).
        fill_value: A numeric scalar value to fill the resulting Tensor.
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to fill_value.

    Example:
    ::

        import numpy as np

        t = tp.Tensor([[1, 2], [3, 4]])
        a = tp.full_like(t, 2)
        assert (a.numpy() == np.array([[2, 2], [2, 2]], dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Fill, fill_value, None, dtype)
