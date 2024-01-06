from dataclasses import dataclass

from tripy import util
from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.base import BaseOperator


@dataclass
class Fill(BaseOperator):
    """
    Represents a fill operation.
    """

    value: float
    shape: ShapeInfo
    dtype: datatype.dtype

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 0, "Fill operation should have no input!"
        assert len(output_names) == 1, "Fill operation should have exactly one output!"
        return f"{output_names[0]} = Tensor.fill(value={self.value}, shape={self.shape}, dtype={self.dtype.name})"

    def infer_shapes(self, input_shapes):
        return [util.make_tuple(self.shape)]

    def infer_dtypes(self, input_dtypes):
        return [self.dtype]

    def infer_devices(self, input_devices):
        from tripy.common import device

        return [device("gpu")]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        import numpy as np
        from tripy.flat_ir.ops import BroadcastOp, ConstantOp

        const_val_tensor = flat_ir.add_tensor(outputs[0], shape=[])
        const_val_op = ConstantOp(self, [], [const_val_tensor], data=np.array(self.value, dtype=self.dtype.name))

        flat_ir.ops.append(const_val_op)
        flat_ir.ops.append(BroadcastOp(self, [const_val_tensor], outputs, broadcast_dim=[]))


def ones(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32):
    """
    Creates a Tensor with all elements set to 1.

    Args:
        shape: A list or tuple of integers
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 1.

    Example:
    ::

        import numpy as np

        a = tp.ones([2, 3])
        assert (a.numpy() == np.ones([2, 3], dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([], Fill(1, shape, dtype))


def zeros(shape: ShapeInfo, dtype: datatype.dtype = datatype.float32):
    """
    Creates a Tensor with all elements set to 0.

    Args:
        shape: A list or tuple of integers
        dtype: Optional datatype of an element in the resulting Tensor.

    Returns:
        A Tensor with all elements set to 0.

    Example:
    ::

        import numpy as np

        a = tp.zeros([2, 3])
        assert (a.numpy() == np.zeros([2, 3], dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([], Fill(0, shape, dtype))
