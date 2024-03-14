from dataclasses import dataclass

import tripy.frontend.trace.ops.utils as op_utils
from tripy import utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import export


@dataclass(repr=False)
class Unsqueeze(BaseTraceOp):

    dim: int

    def infer_shapes(self):
        assert len(self.inputs) == 1, "Unsqueeze operation should have exactly one input!"
        if self.dim < 0:
            self.dim += len(self.inputs[0].shape) + 1

        out_shape = list(self.inputs[0].shape)
        out_shape.insert(self.dim, 1)
        self.outputs[0].shape = utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):

        from tripy.common.array import Array
        from tripy.common.datatype import int32
        from tripy.common.device import device
        from tripy.flat_ir.ops import ConcatenateOp, ConstantOp, DynamicBroadcastOp, SliceOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.frontend.dim import Dim

        broadcast_dim = list(range(len(inputs[0].shape)))
        for idx in range(len(broadcast_dim)):
            if idx >= self.dim:
                broadcast_dim[idx] += 1

        # Get the shape of input, insert 1 at the required index via slicing and concatenating.
        # Use dynamic shape broadcast op and provide the new shape tensor as the output shape

        # Get the input shape
        shape_output_tensor = op_utils.get_shape_of_tensor(inputs[0])

        # Create a constant of Dim[1] filled with 1
        const_output_tensor = FlatIRTensor.build(
            shape=(Dim(1),),
            dtype=int32,
            device=inputs[0].device,
            reason_details=[
                "create a constant value tensor containing a '1' to concatenate with "
                "the shape of the input tensor to create the 'unsqueeze' output shape"
            ],
        )
        ConstantOp.build(
            [],
            [const_output_tensor],
            data=Array([1], int32, shape=(1,), device=device("cpu")),
        )

        # Slice the first half of shape : shape[:dim]
        slice_first_half = FlatIRTensor.build(
            shape=(Dim(self.dim),),
            dtype=int32,
            device=inputs[0].device,
            reason_details=[
                f"slice the shape of the input tensor of 'unsqueeze' up to dimension {self.dim}. Note: The input tensor was: ",
                inputs[0],
            ],
        )
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
            shape=(Dim(len(inputs[0].shape) - self.dim),),
            dtype=int32,
            device=inputs[0].device,
            reason_details=[
                f"slice the shape of the input tensor of 'unsqueeze' after dimension {self.dim}. Note: The input tensor was: ",
                inputs[0],
            ],
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
        concat_output_tensor = FlatIRTensor.build(
            shape=(Dim(1 + len(inputs[0].shape)),),
            dtype=int32,
            device=inputs[0].device,
            reason_details=[
                "concatenate the input shape with 1s to create the output shape of the 'unsqueeze' operation"
            ],
        )

        ConcatenateOp.build([slice_first_half, const_output_tensor, slice_second_half], [concat_output_tensor], dim=0)

        DynamicBroadcastOp.build(
            [inputs[0], concat_output_tensor],
            [outputs[0]],
            broadcast_dim=broadcast_dim,
        )


@export.public_api(document_under="tensor")
def unsqueeze(input: "tripy.Tensor", dim: int) -> "tripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor with a
    singleton dimension inserted at the specified position.

    Args:
        input: The input tensor.
        dim: The index before which to insert the singleton dimension.

    Returns:
        A new tensor of the same data type as the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota((2, 2), dtype=tp.float32)
        output = tp.unsqueeze(input, 1)

        assert np.array_equal(output.numpy(), np.expand_dims(input.numpy(), 1))
    """
    from tripy.frontend import Tensor

    return Tensor.build([input], Unsqueeze, dim)
