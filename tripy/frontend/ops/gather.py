from dataclasses import dataclass

import tripy.frontend.ops.utils as op_utils
from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.utils import to_dims


@dataclass
class Gather(BaseOperator):
    """
    Represents a gather operation.
    """

    axis: int

    def infer_shapes(self):
        data_shape = self.inputs[0].shape
        indices_shape = self.inputs[1].shape

        out_shape = data_shape[: self.axis] + indices_shape + data_shape[self.axis + 1 :]
        self.outputs[0].shape = to_dims(out_shape)

    def infer_dtypes(self):
        from tripy import int32

        if self.inputs[1].dtype != int32:
            op_utils.raise_error_io_info(
                self,
                "Index tensor for gather operation should be of int32 type.",
                details=[
                    f"Input tensor 'index' for operation: 'gather' must be of int32 type, but 'index' has type: {self.inputs[1].dtype}."
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

        GatherOp(self, inputs, outputs, self.axis)


@TENSOR_METHOD_REGISTRY("gather")
def gather(self: "tripy.Tensor", dim: int, index: "tripy.Tensor") -> "tripy.Tensor":
    """
    Gather values from this tensor using the indices provided along an axis.
    Note that this op behaves similar to numpy take operation.

    Args:
        index: The indices of elements to gather.
        dim: Axis along which data is gathered.

    Returns:
        Data gathered from input tensor.

    Example:

    .. code:: python
        :number-lines:

        data = tp.iota((3,2,2))
        indices = tp.arange(0, 3, dtype=tp.int32)
        out = data.gather(0, indices)

        print(f"data : {data}")
        print(f"index : {indices}")
        print(f"output : {out}")

        assert np.array_equal(out.numpy(), np.take(data.numpy(), indices.numpy(), axis=0))
    """
    from tripy.frontend import Tensor

    out = Tensor.build([self, index], Gather, dim)
    return out
