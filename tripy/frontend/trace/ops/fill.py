import numbers
from dataclasses import dataclass
from typing import Optional

from tripy import export, utils
from tripy.common import datatype
from tripy.common.types import ShapeInfo
from tripy.common.utils import is_supported_array_type
from tripy.frontend.trace.ops.base import BaseTraceOp
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Fill(BaseTraceOp):
    value: float
    shape: ShapeInfo
    dtype: datatype.dtype
    is_input_shape_tensor: bool = False

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def infer_rank(self):
        if self.is_input_shape_tensor:
            from tripy.backend.mlir.utils import ShapeContext

            out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[0])
            assert len(out_shape) == 1, f"Expected rank of shape tensor to be 1, got {len(out_shape)}"
            assert (
                out_shape[0] > 0
            ), f"Incorrect shape of shape tensor, expected shape to be positive, got {out_shape[0]}"
            self.inputs[0].shape = utils.to_dims(out_shape)
            self.outputs[0].rank = self.inputs[0].shape[0].runtime_value
        else:
            self.outputs[0].rank = len(self.shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.common.array import Array
        from tripy.common.device import device
        from tripy.flat_ir.ops import ConstantOp, DynamicBroadcastOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.frontend.tensor import convert_list_data_to_array
        from tripy.frontend.trace.ops.cast import cast
        import tripy.common.datatype as datatype

        const_val_tensor = FlatIRTensor.build(
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor (containing {self.value}) for a fill operation"],
        )
        if not is_supported_array_type(self.dtype):
            data = convert_list_data_to_array(self.value, shape=(), dtype=self.dtype, device=device("cpu"))
        else:
            data = Array(self.value, self.dtype, shape=(), device=device("cpu"))
        ConstantOp.build([], [const_val_tensor], data=data)
        if len(inputs) == 1:
            if self.is_input_shape_tensor:
                output_shape = inputs[0]
            else:
                # Used for FillLike where the shape of output is provided by another tensor.
                output_shape = op_utils.get_shape_of_tensor(inputs[0])
        else:
            output_shape = op_utils.add_constant_tensor_from_list(
                [s.runtime_value for s in self.shape], device=outputs[0].device
            )

        assert output_shape.rank == 1
        DynamicBroadcastOp.build(
            [const_val_tensor, output_shape],
            outputs,
            broadcast_dim=[],
        )


@dataclass(repr=False)
class FillLike(Fill):

    def infer_dtypes(self):
        if self.dtype is None:
            self.dtype = self.inputs[0].dtype

        super().infer_dtypes()

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank


@export.public_api(document_under="tensor_operations")
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

        assert np.array_equal(cp.from_dlpack(output).get(), np.full([2, 3], 2, dtype=np.float32))
    """
    from tripy.frontend import Shape

    if isinstance(shape, Shape):
        return Fill.build([shape], value, None, dtype, True)

    return Fill.build([], value, utils.to_dims(shape), dtype)


@export.public_api(document_under="tensor_operations")
def full_like(input: "tripy.Tensor", value: numbers.Number, dtype: Optional["tripy.dtype"] = None) -> "tripy.Tensor":
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

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[2, 2], [2, 2]], dtype=np.float32))
    """
    return FillLike.build([input], value, None, dtype, False)
