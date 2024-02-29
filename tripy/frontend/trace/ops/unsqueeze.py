from dataclasses import dataclass

from tripy import utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
import tripy.frontend.trace.ops.utils as op_utils


@dataclass(repr=False)
class Unsqueeze(BaseTraceOp):
    """
    Represents an unsqueeze operation.
    """

    dim: int

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Unsqueeze operation should have exactly one input!"
        if self.dim < 0:
            self.dim += len(self.inputs[0].shape) + 1

        out_shape = list(self.inputs[0].shape)
        out_shape.insert(self.dim, 1)
        self.outputs[0].shape = utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import BroadcastOp, ConcatenateOp, DynamicBroadcastOp, SliceOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import int32
        from tripy.frontend.dim import Dim

        broadcast_dim = list(range(len(inputs[0].shape)))
        for idx in range(len(broadcast_dim)):
            if idx >= self.dim:
                broadcast_dim[idx] += 1

        # Get the shape of input, insert 1 at the required index via slicing and concatenating.
        # Use dynamic shape broadcast op and provide the new shape tensor as the output shape

        # Get the input shape
        shape_output_tensor = op_utils.get_shape_of_tensor(inputs[0])

        # # Create a constant of Dim[1] filled with 1
        const_output_tensor = op_utils.add_constant_tensor_from_list([1], inputs[0].device)

        concat_output_tensor = FlatIRTensor.build(
            shape=(Dim(1 + len(inputs[0].shape)),), dtype=int32, device=inputs[0].device
        )

        # Slice the first half of shape : shape[:dim]
        slice_first_half = FlatIRTensor.build(shape=(Dim(self.dim),), dtype=int32, device=inputs[0].device)
        start_indices, limit_indices, strides = op_utils.get_slice_indices(
            self, shape_output_tensor.shape, (slice(0, self.dim, None),)
        )

        SliceOp.build(
            [shape_output_tensor],
            [slice_first_half],
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        )

        # Slice the second half of shape : shape[dim:]
        slice_second_half = FlatIRTensor.build(
            shape=(Dim(len(inputs[0].shape) - self.dim),), dtype=int32, device=inputs[0].device
        )

        start_indices, limit_indices, strides = op_utils.get_slice_indices(
            self, shape_output_tensor.shape, (slice(self.dim, None, None),)
        )

        SliceOp.build(
            [shape_output_tensor],
            [slice_second_half],
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        )

        # concatenate [slice_first_half, 1, slice_second_half]
        ConcatenateOp.build([slice_first_half, const_output_tensor, slice_second_half], [concat_output_tensor], dim=0)

        DynamicBroadcastOp.build(
            [inputs[0], concat_output_tensor],
            [outputs[0]],
            broadcast_dim=broadcast_dim,
        )


@TENSOR_METHOD_REGISTRY("unsqueeze")
def unsqueeze(self, dim: int) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of this tensor with a
    singleton dimension inserted at the specified position.

    Args:
        dim: The index before which to insert the singleton dimension.

    Returns:
        A new tensor of the same data type as this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.ones((2, 2), dtype=tp.float32)
        output = input.unsqueeze(1)

        assert np.array_equal(output.numpy(), np.expand_dims(input.numpy(), 1))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self], Unsqueeze, dim)
