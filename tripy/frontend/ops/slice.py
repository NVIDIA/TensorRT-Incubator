import math
from dataclasses import dataclass
from typing import Tuple, Union

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops.utils import to_dims, get_slice_indices
from tripy.utils import make_tuple


@dataclass
class Slice(BaseOperator):
    """
    Represents a slice operation.
    """

    index: Tuple[Union[slice, int]]

    def infer_shapes(self):
        input_shape = self.inputs[0].shape
        self.start_indices, self.limit_indices, self.strides = get_slice_indices(input_shape, self.index)
        out_shape = [
            math.ceil((stop - start) / stride)
            for start, stop, stride in zip(self.start_indices, self.limit_indices, self.strides)
        ]
        self.outputs[0].shape = to_dims(out_shape)

    def to_flat_ir(self, flat_ir):
        from tripy.flat_ir.ops import SliceOp

        if any(dim.is_dynamic_dim() for dim in self.inputs[0].shape):
            raise NotImplementedError("Dynamic slice is not supported")

        flat_ir.add_op(
            self,
            SliceOp,
            self.inputs,
            self.outputs,
            start_indices=self.start_indices,
            limit_indices=self.limit_indices,
            strides=self.strides,
        )


@TENSOR_METHOD_REGISTRY("__getitem__")
def get_item(self: "tripy.Tensor", index: "index_expr"):
    """
    Returns a tensor that is sliced from the input Tensor.

    Args:
        index: index expression to slice the tensor

    Returns:
        the sliced Tensor

    Example:
    ::

        a = tp.arange(6, dtype=tp.float32).reshape((1, 2, 3, 1))
        out = a[:, 1:2, :-1, 0]
        assert np.array_equal(out.numpy(), np.arange(6, dtype=np.float32).reshape((1, 2, 3, 1))[:, 1:2, :-1, 0])
    """
    from tripy.frontend import Tensor

    index = make_tuple(index)
    out = Tensor.build([self], Slice, index)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = out.squeeze(make_tuple(squeeze_dims))

    return out
