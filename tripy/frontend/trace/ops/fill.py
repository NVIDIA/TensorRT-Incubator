import numbers
from dataclasses import dataclass

from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy import utils


@dataclass(repr=False)
class Fill(BaseTraceOp):
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
        from tripy.flat_ir.tensor import FlatIRTensor

        const_val_tensor = FlatIRTensor.build(shape=[], dtype=outputs[0].dtype, device=outputs[0].device)
        ConstantOp(self, [], [const_val_tensor], data=np.array(self.value, dtype=self.dtype.name))
        BroadcastOp(self, [const_val_tensor], outputs, broadcast_dim=[])


@dataclass(repr=False)
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


def full(shape: ShapeInfo, value: numbers.Number, dtype: "tripy.dtype" = datatype.float32) -> "tripy.Tensor":
    """
    Returns a tensor of the desired shape with all values set to the specified value.

    Args:
        shape: The desired shape.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape`` and data type ``dtype``.

    .. code-block:: python
        :linenos:
        :caption: Example

        output = tp.full(shape=[2, 3], value=2)

        assert np.array_equal(output.numpy(), np.full([2, 3], 2, dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([], Fill, value, utils.to_dims(shape), dtype)


def full_like(input: "tripy.Tensor", value: numbers.Number, dtype: "tripy.dtype" = None) -> "tripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with all values
    set to the specified value.

    Args:
        input: Input tensor.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        output = tp.full_like(input, value=2)

        assert np.array_equal(output.numpy(), np.array([[2, 2], [2, 2]], dtype=np.float32))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], FillLike, value, None, dtype)