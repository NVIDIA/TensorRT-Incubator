from dataclasses import dataclass
from typing import Optional, Sequence

from tripy import export, utils
from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Iota(BaseTraceOp):
    dim: int
    output_rank: int
    dtype: datatype.dtype

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_rank(self):
        if self.output_rank is None:
            if self.inputs[0].shape is None:
                from tripy.backend.mlir.utils import ShapeContext

                out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[0])
                assert len(out_shape) == 1
                assert out_shape[0] >= 0, f"incorrect shape computation {out_shape}"
                self.output_rank = out_shape[0]
            else:
                self.output_rank = self.inputs[0].shape[0].runtime_value

        # Iota requires inputs[0] to be statically shaped
        if self.inputs[0].shape is None:
            self.inputs[0].shape = utils.to_dims((self.output_rank,))

        if self.dim < 0:
            self.dim += self.output_rank
        self.outputs[0].rank = self.output_rank

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicIotaOp

        DynamicIotaOp.build(inputs, outputs, dim=self.dim)


@frontend_utils.convert_inputs_to_tensors(exclude=["dim", "dtype", "output_rank"], shape_argument=["shape"])
def iota_impl(shape: ShapeInfo, dim: int, dtype: datatype.dtype, output_rank: int) -> "tripy.Tensor":
    return Iota.build([shape], dim, output_rank, dtype)


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
    output_rank = len(shape) if isinstance(shape, Sequence) else None
    return iota_impl(shape, dim, dtype, output_rank)


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
    return iota_impl(input.shape, dim, utils.default(dtype, input.dtype), input.rank)
