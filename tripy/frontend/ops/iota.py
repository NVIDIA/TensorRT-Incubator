from dataclasses import dataclass

from tripy.common import datatype
from tripy.common.exception import raise_error
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

    def infer_shapes(self):
        self.outputs[0].shape = self.shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import IotaOp

        IotaOp(self, inputs, outputs, dim=self.dim)


@dataclass
class IotaLike(Iota):
    """
    Represents an iota_like operation.
    """

    def infer_shapes(self):
        self.shape = self.inputs[0].shape
        super().infer_shapes()

    def infer_dtypes(self):
        if self.dtype is None:
            self.dtype = self.inputs[0].dtype
        super().infer_dtypes()


def iota(shape: ShapeInfo, dim: int = 0, dtype: datatype.dtype = datatype.float32):
    """
    Fills an output tensor with values in increasing order starting from zero along the given dimension

    Args:
        shape: Shape of the resulting Tensor
        dim: Dimension of the iota operation
             constraint: 0 <= dim < rank(shape)
        dtype: Optional dtype of the resulting Tensor

    Example:

    .. code:: python
        :number-lines:

        a = tp.iota([3])
        print(a)
        assert np.array_equal(a.numpy(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend import Tensor

    if dim < 0 or dim >= len(shape):
        raise_error(
            "Invalid iota dim.",
            details=[
                "iota dim must be satisfy 0 <= dim < rank(shape), got dim=",
                dim,
                ", while rank of shape is ",
                len(shape),
            ],
        )
    return Tensor.build([], Iota, dim, to_dims(shape), dtype)


def iota_like(input: "tripy.Tensor", dim: int = 0, dtype: datatype.dtype = None):
    """
    Fills an output tensor with values in increasing order starting from zero along the given dimension.
    The output tensor's shape (and dtype if not given) are determined by the input.

    Args:
        input: Input tensor to determine shape (and dtype if not given) of the resulting Tensor
        dim: Dimension of the iota operation
             constraint: 0 <= dim < rank(input_shape)
        dtype: Optional dtype of the resulting Tensor

    Example:

    .. code:: python
        :number-lines:

        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        print(a)
        assert np.array_equal(a.numpy(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], IotaLike, dim, None, dtype)
