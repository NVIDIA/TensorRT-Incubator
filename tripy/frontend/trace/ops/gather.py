from dataclasses import dataclass

import tripy.frontend.trace.ops.utils as op_utils
from tripy import utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops.base import BaseTraceOp


@dataclass(repr=False)
class Gather(BaseTraceOp):
    axis: int

    def infer_shapes(self):
        data_shape = self.inputs[0].shape
        indices_shape = self.inputs[1].shape

        out_shape = data_shape[: self.axis] + indices_shape + data_shape[self.axis + 1 :]
        self.outputs[0].shape = utils.to_dims(out_shape)

    def infer_dtypes(self):
        from tripy import int32

        if self.inputs[1].dtype != int32:
            op_utils.raise_error_io_info(
                self,
                "Index tensor for gather operation should be of int32 type.",
                details=[
                    f"Input tensor 'index' for operation: 'gather' must be of int32 type, but 'index' has type: {self.inputs[1].dtype}"
                ],
            )

        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_devices(self):
        from tripy.common import device

        self.outputs[0].device = device("gpu")

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import GatherOp

        if any(dim.is_dynamic_dim() for dim in (inputs[0].shape + inputs[1].shape)):
            raise NotImplementedError("Dynamic gather is not supported")

        GatherOp.build(inputs, outputs, self.axis)


@TENSOR_METHOD_REGISTRY("gather")
def gather(self, dim: int, index: "tripy.Tensor") -> "tripy.Tensor":
    """
    Gather values from this tensor along the specified axis based on the specified indices.
    This behaves similarly to ``numpy.take()``.

    Args:
        dim: Axis along which data is gathered.
        index: The indices of elements to gather.

    Returns:
        A new tensor of the same data type as this tensor and same shape along every
        dimension except ``dim``, which will have a size equal to ``len(index)``.

    .. code-block:: python
        :linenos:
        :caption: Example

        data = tp.iota((3, 2, 2))
        indices = tp.Tensor([0, 2], dtype=tp.int32)
        output = data.gather(0, indices)

        assert np.array_equal(output.numpy(), np.take(data.numpy(), indices.numpy(), axis=0))
    """
    from tripy.frontend import Tensor

    return Tensor.build([self, index], Gather, dim)
