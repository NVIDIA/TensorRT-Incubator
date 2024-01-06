from dataclasses import dataclass

from tripy import util
from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.ops.base import BaseOperator


@dataclass
class Iota(BaseOperator):
    """
    Represents an iota operation.
    """

    dim: int
    shape: ShapeInfo
    dtype: datatype.dtype

    def to_trace_str(self, input_names, output_names):
        assert len(input_names) == 0, "Iota operation should have no input!"
        assert len(output_names) == 1, "Iota operation should have exactly one output!"
        return f"{output_names[0]} = Tensor.iota(dim={self.dim}, shape={self.shape}, dtype={self.dtype.name})"

    def infer_shapes(self, input_shapes):
        return [util.make_tuple(self.shape)]

    def infer_dtypes(self, input_dtypes):
        return [self.dtype]

    def infer_devices(self, input_devices):
        from tripy.common import device

        return [device("gpu")]

    def to_flat_ir(self, flat_ir, inputs, outputs):
        from tripy.flat_ir.ops import IotaOp

        flat_ir.ops.append(IotaOp(self, inputs, outputs, dim=self.dim))


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
    return Tensor.build([], Iota(dim, shape, dtype))
