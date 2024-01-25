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

    def infer_shapes(self):
        self.outputs[0].shape = self.shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        import numpy as np

        from tripy.flat_ir.ops import BroadcastOp, ConstantOp
        from tripy.flat_ir.tensor import FIRTensor

        const_val_tensor = FIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)
        ConstantOp(self, [], [const_val_tensor], data=np.array(self.value, dtype=self.dtype.name))
        BroadcastOp(self, [const_val_tensor], outputs, broadcast_dim=[])


@dataclass
class FillLike(Fill):
    """
    Represents a fill_like operation.
    """

    def infer_shapes(self):
        self.shape = self.inputs[0].shape
        super().infer_shapes()

    def infer_dtypes(self):
        if self.dtype is None:
            self.dtype = self.inputs[0].dtype
        super().infer_dtypes()


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

    .. code:: python
        :number-lines:

        a = tp.full([2, 3], 2)
        print(a)
        assert np.array_equal(a.numpy(), np.full([2, 3], 2, dtype=np.float32))
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

    .. code:: python
        :number-lines:

        t = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        a = tp.full_like(t, 2)
        print(a)
        assert np.array_equal(a.numpy(), np.array([[2, 2], [2, 2]], dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], FillLike, fill_value, None, dtype)
