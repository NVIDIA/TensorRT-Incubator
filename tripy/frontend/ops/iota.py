from dataclasses import dataclass

from tripy import util
from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.utils import to_dims


@dataclass
class Iota(BaseOperator):
    """
    Represents an iota operation.
    """

    dim: int
    shape: ShapeInfo
    dtype: datatype.dtype

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 0 or len(input_names) == 1, "Iota operation should have 0 or 1 input!"
        assert len(output_names) == 1, "Iota operation should have exactly one output!"
        if len(input_names) == 1:
            return f"{output_names[0]} = Tensor.iota(dim={self.dim}, like={input_names[0]})"
        else:
            return f"{output_names[0]} = Tensor.iota(dim={self.dim}, shape={self.shape}, dtype={self.dtype.name})"

    def infer_shapes(self):
        if len(self.inputs) == 1:
            self.shape = self.inputs[0].shape
        self.outputs[0].shape = self.shape

    def infer_dtypes(self, input_dtypes):
        if len(input_dtypes) == 1 and self.dtype is None:
            self.dtype = input_dtypes[0]
        return [self.dtype]

    def infer_devices(self, input_devices):
        from tripy.common import device

        return [device("gpu")]

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import IotaOp

        flat_ir.add_op(self, IotaOp, self.inputs, self.outputs, dim=self.dim)


def arange(shape: ShapeInfo, dim: int = 0, dtype: datatype.dtype = datatype.float32):
    """
    Fills an output tensor with values in increasing order starting from zero along the given dimension

    Args:
        shape: Shape of the resulting Tensor
        dim: Dimension of the iota operation
             constraint: 0 <= dim < rank(shape)
        dtype: Optional dtype of the resulting Tensor

    Example:
    ::

        import numpy as np

        a = tp.arange([3])
        assert (a.numpy() == np.arange(0, 3, dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    if dim < 0 or dim >= len(shape):
        raise Exception("Invalid arange dim")
    return Tensor.build([], Iota, dim, to_dims(shape), dtype)


def arange_like(input: "tripy.Tensor", dim: int = 0, dtype: datatype.dtype = None):
    """
    Fills an output tensor with values in increasing order starting from zero along the given dimension.
    The output tensor's shape (and dtype if not given) are determined by the input.

    Args:
        input: Input tensor to determine shape (and dtype if not given) of the resulting Tensor
        dim: Dimension of the iota operation
             constraint: 0 <= dim < rank(input_shape)
        dtype: Optional dtype of the resulting Tensor

    Example:
    ::

        import numpy as np

        t = tp.Tensor([1, 2, 3])
        a = tp.arange_like(t)
        assert (a.numpy() == np.arange(0, 3, dtype=np.float32)).all()
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Iota, dim, None, dtype)
