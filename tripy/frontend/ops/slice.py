import math
from dataclasses import dataclass
from typing import Tuple, Union

from tripy.frontend.ops.base import BaseOperator
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.ops import utils as op_utils
from tripy.utils import make_tuple

import copy


@dataclass
class Slice(BaseOperator):
    """
    Represents a slice operation.
    """

    index: Tuple[Union[slice, int]]

    def get_slice_indices(self, shape):
        """
        Converts index to slices required by Slice operation

        Args:
            shape: shape of input tensor

        Returns:
            start_indices: list of start slice index
            limit_indices: list of end slice index
            strides: list of slice strides
        """
        # TODO: only works for static shape, figure out how to handle DS
        runtime_shape = [dim.runtime_value for dim in shape]

        index = copy.copy(self.index)

        dims = len(shape)
        if len(index) > dims:
            op_utils.raise_error_io_info(
                self,
                "Too many indices for input tensor.",
                details=[
                    "Input tensor has a rank of ",
                    dims,
                    " but was attempted to be sliced with ",
                    len(index),
                    " indices.",
                ],
            )
        index += (dims - len(index)) * (slice(None),)
        start_indices = []
        limit_indices = []
        strides = []
        to_positive_idx = lambda idx, dim: idx + dim if idx < 0 else idx
        for idx, dim in zip(index, runtime_shape):
            if isinstance(idx, int):
                # slice the single element and squeeze later
                idx = to_positive_idx(idx, dim)
                start_indices.append(idx)
                limit_indices.append(idx + 1)
                strides.append(1)
            else:
                start_indices.append(to_positive_idx(idx.start, dim) if idx.start else 0)
                limit_indices.append(to_positive_idx(idx.stop, dim) if idx.stop else dim)
                strides.append(idx.step if idx.step else 1)
        return start_indices, limit_indices, strides

    def infer_shapes(self):
        input_shape = self.inputs[0].shape
        self.start_indices, self.limit_indices, self.strides = self.get_slice_indices(input_shape)
        out_shape = [
            math.ceil((stop - start) / stride)
            for start, stop, stride in zip(self.start_indices, self.limit_indices, self.strides)
        ]
        self.outputs[0].shape = op_utils.to_dims(out_shape)

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import SliceOp

        if any(dim.is_dynamic_dim() for dim in inputs[0].shape):
            raise NotImplementedError("Dynamic slice is not supported")

        SliceOp(
            self,
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

        input = tp.arange(6, dtype=tp.float32).reshape((1, 2, 3, 1))
        output = input[:, 1:2, :-1, 0]

        assert np.array_equal(output.numpy(), np.arange(6, dtype=np.float32).reshape((1, 2, 3, 1))[:, 1:2, :-1, 0])
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
