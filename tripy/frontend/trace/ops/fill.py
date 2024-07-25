import numbers
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import tripy.frontend.trace.ops.utils as op_utils
import tripy.frontend.utils as frontend_utils
from tripy import export, utils
from tripy.common import datatype
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Fill(BaseTraceOp):
    value: float
    output_rank: int
    dtype: datatype.dtype

    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.never_return_shape

    def infer_dtypes(self):
        self.outputs[0].dtype = self.dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def infer_rank(self):
        if self.output_rank is None:
            if self.inputs[0].shape is None:
                from tripy.backend.mlir.utils import ShapeContext

                out_shape = ShapeContext().get_shape_of_dynamic_trace_tensor(self.inputs[0])
                assert len(out_shape) == 1, f"Expected rank of shape tensor to be 1, got {len(out_shape)}"
                assert (
                    out_shape[0] >= 0
                ), f"Incorrect shape of shape tensor, expected shape to be positive, got {out_shape[0]}"
                self.inputs[0].shape = out_shape
            self.output_rank = self.inputs[0].shape[0]
        self.outputs[0].rank = self.output_rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.common.array import Array
        from tripy.common.device import device
        from tripy.flat_ir.ops import ConstantOp, DynamicBroadcastOp
        from tripy.flat_ir.tensor import FlatIRTensor

        const_val_tensor = FlatIRTensor.build(
            rank=0,
            dtype=outputs[0].dtype,
            device=outputs[0].device,
            reason_details=[f"create the constant value tensor (containing {self.value}) for a fill operation"],
        )
        data = Array(self.value, shape=(), dtype=self.dtype, device=device("cpu"))
        ConstantOp.build([], [const_val_tensor], data=data)

        DynamicBroadcastOp.build(
            [const_val_tensor, inputs[0]],
            outputs,
            broadcast_dim=[],
        )


@frontend_utils.convert_inputs_to_tensors(exclude=["value", "dtype", "output_rank"], shape_argument=["shape"])
def full_impl(shape: Tuple[int], value: numbers.Number, dtype: "tripy.dtype", output_rank: int) -> "tripy.Tensor":
    return Fill.build([shape], value, output_rank, dtype)


@export.public_api(document_under="tensor_operations")
@frontend_utils.convert_inputs_to_tensors(shape_argument=["shape"], exclude=["value", "dtype"])
def full(
    shape: Union["tripy.Shape", Sequence[int]], value: numbers.Number, dtype: "tripy.dtype" = datatype.float32
) -> "tripy.Tensor":
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
    output_rank = len(shape) if isinstance(shape, Sequence) else None
    return full_impl(shape, value, dtype, output_rank)


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
    return full_impl(input.shape, value, utils.default(dtype, input.dtype), input.rank)
