from dataclasses import dataclass
from typing import Optional

from tripy import export, utils
from tripy.common import datatype
from tripy.common.exception import raise_error
from tripy.common.types import ShapeInfo
from tripy.common.utils import is_supported_array_type
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Iota(BaseTraceOp):
    dim: int
    shape: ShapeInfo
    dtype: datatype.dtype

    def infer_rank(self):
        if self.dim < 0:
            self.dim += len(self.shape)
        self.outputs[0].rank = len(self.shape)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicIotaOp

        output_shape = [s.runtime_value for s in self.shape]
        DynamicIotaOp.build(
            [op_utils.add_constant_tensor_from_list(output_shape, outputs[0].device)], outputs, dim=self.dim
        )


@dataclass(repr=False)
class IotaLike(Iota):
    """
    Represents an iota_like operation.
    """

    def infer_rank(self):
        if self.dim < 0:
            self.dim += self.inputs[0].rank
        self.outputs[0].rank = self.inputs[0].rank

    def infer_dtypes(self):
        if self.dtype is None:
            self.dtype = self.inputs[0].dtype
        super().infer_dtypes()

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicIotaOp

        DynamicIotaOp.build([op_utils.get_shape_of_tensor(inputs[0])], outputs, dim=self.dim)


@export.public_api(document_under="tensor_operations")
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

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend.trace.ops.cast import cast

    if not is_supported_array_type(dtype):
        result = Iota.build([], dim, utils.to_dims(shape), datatype.float32)
        return cast(result, dtype)

    return Iota.build([], dim, utils.to_dims(shape), dtype)


@export.public_api(document_under="tensor_operations")
def iota_like(input: "tripy.Tensor", dim: int = 0, dtype: Optional[datatype.dtype] = None) -> "tripy.Tensor":
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

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    from tripy.frontend.trace.ops.cast import cast

    result_dtype = dtype if dtype is not None else input.dtype
    if result_dtype not in (datatype.float32, datatype.int32):
        result = IotaLike.build([input], dim, None, datatype.float32)
        return cast(result, result_dtype)

    return IotaLike.build([input], dim, None, dtype)
