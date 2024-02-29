from dataclasses import dataclass

from tripy import utils
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Iota(BaseTraceOp):
    """
    Represents an iota operation.
    """

    dim: int
    shape: ShapeInfo
    dtype: datatype.dtype

    def infer_shapes(self):
        self.outputs[0].shape = self.shape
        if self.dim < 0:
            self.dim += len(self.shape)

        if self.dim < 0 or self.dim >= len(self.shape):
            raise_error(
                "Invalid iota dim.",
                details=[
                    "iota dim must be satisfy 0 <= dim < rank(shape), got dim=",
                    self.dim,
                    ", while rank of shape is ",
                    len(self.shape),
                ],
            )

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import IotaOp

        IotaOp.build(inputs, outputs, dim=self.dim)


@dataclass(repr=False)
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


def iota(shape: ShapeInfo, dim: int = 0, dtype: datatype.dtype = datatype.float32) -> "tripy.Tensor":
    """
    Fills an output tensor with consecutive values starting from zero along the given dimension.

    Args:
        shape: The desired shape.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape`` and data type ``dtype``.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.iota((3,), dim=-1)

        assert np.array_equal(output.numpy(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([], Iota, dim, utils.to_dims(shape), dtype)


def iota_like(input: "tripy.Tensor", dim: int = 0, dtype: datatype.dtype = None) -> "tripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with consecutive values
    starting from zero along the given dimension.

    Args:
        input: Input tensor.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1, 2, 3])
        output = tp.iota_like(input)

        assert np.array_equal(output.numpy(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], IotaLike, dim, None, dtype)
