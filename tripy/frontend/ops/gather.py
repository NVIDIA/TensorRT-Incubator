import copy
from dataclasses import dataclass

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.utils import to_dims
import tripy.frontend.ops.utils as op_utils


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

    def to_flat_ir(self, flat_ir):
        import tripy.flat_ir.utils as flat_ir_utils
        from tripy.flat_ir.ops import GatherOp

        if any(dim.is_dynamic_dim() for dim in (self.inputs[0].shape + self.inputs[1].shape)):
            raise NotImplementedError("Dynamic gather is not supported")

        inputs = copy.copy(self.inputs)
        # Reshape indices and add extra dimension at the end (required for stablehlo translation)
        index_shape = to_dims(inputs[1].shape + (1,))
        inputs[1] = flat_ir_utils.insert_reshape(self, flat_ir, inputs[1], index_shape)

        flat_ir.add_op(
            self,
            GatherOp,
            self.inputs,
            self.outputs,
            self.axis,
        )


def gather(tensor: "tripy.Tensor", index: "index_expr", axis):
    """
    Gather values from data tensor using the indices provided along an axis.
    Note that this op behaves similar to numpy take operation.

    Args:
        tensor: data tensor to gather data from.
        index: tensor
        axis: axis along which data is gathered.

    Returns:
        data gathered from input tensor.

    Example:
    ::

        data = tp.iota((3,2,2))
        indices = tp.arange(0, 3, dtype=tp.int32)
        out = tp.gather(data, indices, axis=0)

        print(f"data : {data}")
        print(f"index : {indices}")
        print(f"axis : 0")
        print(f"output : {out}")

        assert np.array_equal(out.numpy(), np.take(data.numpy(), indices.numpy(), axis=0))
    """
    from tripy.frontend import Tensor

    out = Tensor.build([tensor, index], Gather, axis)
    return out
