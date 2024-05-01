from dataclasses import dataclass

from tripy import export, utils
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Unsqueeze(BaseTraceOp):

    dim: int

    def infer_shapes(self):
        if self.dim < 0:
            self.dim += len(self.inputs[0].shape) + 1

        out_shape = list(self.inputs[0].shape)
        out_shape.insert(self.dim, 1)
        self.outputs[0].shape = utils.to_dims(out_shape)

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):
        self.outputs[0].rank = self.inputs[0].rank + 1

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicBroadcastOp

        broadcast_dim = list(range(len(inputs[0].shape)))
        for idx in range(len(broadcast_dim)):
            if idx >= self.dim:
                broadcast_dim[idx] += 1

        DynamicBroadcastOp.build(
            [inputs[0], inputs[1]],
            [outputs[0]],
            broadcast_dim=broadcast_dim,
        )


# Two operand unsqueeze op to ensure that Trace op is 1:1 with Python code (for error messaging).
def unsqueeze_two_operand(input, result_shape, dim):
    return Unsqueeze.build([input, result_shape], dim)


@export.public_api(document_under="tensor_operations")
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
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.reshape import reshape

    from tripy.frontend import Tensor

    # Add specical case for rank 0 since tensor.shape is not supported when rank is 0.
    if input.rank == 0:
        result_shape = Tensor([1])
    else:
        input_shape = input.shape
        result_shape = concatenate([input_shape[:dim], Tensor([1]), input_shape[dim:]], dim=0)
    return unsqueeze_two_operand(input, result_shape, dim)
