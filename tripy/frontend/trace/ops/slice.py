import copy
import math
from dataclasses import dataclass
from typing import Tuple, Union

from tripy import utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import make_tuple


@dataclass(repr=False)
class Slice(BaseTraceOp):

    index: Tuple[Union[slice, int]]

    def infer_shapes(self):
        input_shape = self.inputs[0].shape
        self.start_indices, self.limit_indices, self.strides = op_utils.get_slice_indices(self, input_shape, self.index)
        out_shape = [
            math.ceil((stop - start) / stride)
            for start, stop, stride in zip(self.start_indices, self.limit_indices, self.strides)
        ]
        self.outputs[0].shape = utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import SliceOp

        if any(dim.is_dynamic_dim() for dim in inputs[0].shape):
            raise NotImplementedError("Dynamic slice is not supported")

        SliceOp.build(
            inputs,
            outputs,
            start_indices=self.start_indices,
            limit_indices=self.limit_indices,
            strides=self.strides,
        )


@TENSOR_METHOD_REGISTRY("__getitem__")
def __getitem__(self, index: Union[slice, int, Tuple[int]]) -> "tripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        index: The index or slice.

    Returns:
        A tensor cotnaining the slice of ths tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (1, 2, 3, 1))
        output = input[:, 1:2, :-1, 0]

        assert np.array_equal(output.numpy(), np.arange(6, dtype=np.float32).reshape((1, 2, 3, 1))[:, 1:2, :-1, 0])
    """
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.reshape import squeeze

    index = make_tuple(index)
    out = Tensor.build([self], Slice, index)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = squeeze(out, make_tuple(squeeze_dims))

    return out
