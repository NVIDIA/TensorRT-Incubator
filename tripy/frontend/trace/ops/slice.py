import math
from dataclasses import dataclass
from typing import Tuple, Union
from tripy import utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend import utils as frontend_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import make_tuple
from tripy.frontend.trace.ops.binary_elementwise import Comparison
from tripy.common.exception import raise_error


@dataclass(repr=False)
class Slice(BaseTraceOp):

    index: Tuple[Union[slice, int]]

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_shapes(self):
        # Static shape computation of the output rank is wrong since it should reduce the rank in case single element is selected along a dimension.
        input_shape = self.inputs[0].shape
        self.start_indices, self.limit_indices, self.strides = op_utils.get_slice_indices(self, input_shape, self.index)
        out_shape = [
            math.ceil(abs((stop - start) / stride))
            for start, stop, stride in zip(self.start_indices, self.limit_indices, self.strides)
        ]
        self.outputs[0].shape = utils.to_dims(out_shape)

    def infer_rank(self):
        # How can we compute the output rank in the case when start, size, stride tensors are dynamic?
        self.outputs[0].rank = self.inputs[0].rank

    def to_flat_ir(self, inputs, outputs):
        from tripy.flat_ir.ops import DynamicReshapeOp, DynamicSliceOp, MinOp
        from tripy.flat_ir.tensor import FlatIRTensor
        from tripy.common.datatype import int32

        device = inputs[0].device
        zero_1d = op_utils.add_constant_tensor_from_list([0], device)
        one_1d = op_utils.add_constant_tensor_from_list([1], device)

        data_tensor = inputs[0]
        slice_params = inputs[1:]
        input_rank = data_tensor.rank
        input_shape = op_utils.get_shape_of_tensor(data_tensor)

        start_idxs = []
        limit_idxs = []
        stride_idxs = []

        for dim in range(input_rank):
            shape_slice = op_utils.slice_rank1_tensor(
                input_shape,
                dim,
                reason_details=[
                    "slicing the shape tensor ",
                    input_shape,
                    f" to get the dimension with index {dim}",
                ],
            )

            if dim < len(slice_params) // 3:

                def expand_to_rank1(index_tensor):
                    reshape_out = FlatIRTensor.build(
                        shape=utils.to_dims([1]),
                        rank=1,
                        dtype=int32,
                        device=device,
                        reason_details=["reshape index tensor into singleton in case it is () instead of (1,)"],
                    )
                    shape_input = op_utils.add_constant_tensor_from_list([1], device)
                    DynamicReshapeOp.build([index_tensor, shape_input], [reshape_out])
                    return reshape_out

                # the max dimension is clamped
                def clamp(index_tensor):
                    min_out = FlatIRTensor.build(
                        shape=utils.to_dims([1]),
                        rank=1,
                        dtype=int32,
                        device=device,
                        reason_details=["clamping the slice upper bound to the shape dim"],
                    )
                    MinOp.build([index_tensor, shape_slice], [min_out])
                    return min_out

                start_idxs.append(expand_to_rank1(slice_params[3 * dim]))
                limit_idxs.append(clamp(expand_to_rank1(slice_params[3 * dim + 1])))
                stride_idxs.append(expand_to_rank1(slice_params[3 * dim + 2]))
            else:
                start_idxs.append(zero_1d)
                limit_idxs.append(shape_slice)
                stride_idxs.append(one_1d)

        start_index_tensor = op_utils.concatenate_tensors(start_idxs, dim=0)
        limit_index_tensor = op_utils.concatenate_tensors(limit_idxs, dim=0)
        stride_index_tensor = op_utils.concatenate_tensors(stride_idxs, dim=0)
        DynamicSliceOp.build([data_tensor, start_index_tensor, limit_index_tensor, stride_index_tensor], outputs)


@TENSOR_METHOD_REGISTRY("__getitem__")
def __getitem__(self, index: Union[slice, int, Tuple[int], "tripy.Tensor"]) -> "tripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        index: The index (as an int or Tripy tensor) or slice.

    Returns:
        A tensor containing the slice of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (1, 2, 3, 1))
        output = input[:, 1:2, :-1, 0]
        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(6, dtype=np.float32).reshape((1, 2, 3, 1))[:, 1:2, :-1, 0])

    .. code-block:: python
        :linenos:
        :caption: Negative step size

        input = tp.arange(10)
        output = input[8:2:-1]
        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(10)[8:2:-1])

    """
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.flip import flip
    from tripy.frontend.trace.ops.reshape import reshape, squeeze
    from tripy.frontend.trace.ops.where import where

    index = make_tuple(index)
    # Collect args in the order of (start, stop, step) in a flat list, filling in default values if any is missing.
    # For indices that are single ints/tensors, the default stop is start+1 and the default step is 1
    args = []

    # index can be a tuple of just integer, Tensor (ex; a[2] or a[t]) or can be a
    # slice with optional start, stop and step fields set (where the element can be int or Tensor).
    t_shape = self.shape
    flip_dims = []
    for i, idx in enumerate(index):

        def convert_to_positive_idx(index: Union[int, Tensor]) -> Union[int, Tensor]:
            # Base condition for t_shape[i] else the frontend will recurse infinitely.
            if isinstance(index, int):
                return index if index >= 0 else index + t_shape[i]
            else:
                return where(index >= 0, index, reshape(t_shape[i], (1,)) + index)

        if isinstance(idx, int) or isinstance(idx, Tensor):
            args.append(convert_to_positive_idx(idx))
            args.append(convert_to_positive_idx(idx) + 1)
            args.append(1)
        elif isinstance(idx, slice):
            # For negative strides, we must convert the indices for the flipped dimension.
            # For example, l[8:2:-1] starts from index 8 of the original list and proceeds
            # to index 2 of the original list (exclusive). Index 0 in the original list is the final
            # index of the flipped list, so index len(l) - 1. Index 8 is eight indices afterwards,
            # so, len(l) - 1 - 8 in the flipped list. Hence, l[8:2:-1] starts from index len(l) - 9
            # of the flipped list and goes to index len(l) - 3 of the flipped list.
            if idx.step is not None and idx.step < 0:
                flip_dims.append(i)
                # note that if the starting index is past the end of the tensor, slicing clamps it
                args.append(
                    0
                    if idx.start is None
                    else where(idx.start >= t_shape[i], Tensor(0), t_shape[i] - convert_to_positive_idx(idx.start) - 1)
                )
                args.append(t_shape[i] if idx.stop is None else t_shape[i] - convert_to_positive_idx(idx.stop) - 1)
            else:
                args.append(convert_to_positive_idx(utils.default(idx.start, 0)))
                args.append(convert_to_positive_idx(utils.default(idx.stop, t_shape[i])))
            args.append(abs(utils.default(idx.step, 1)))
        else:
            raise_error(
                "Slice index type is not supported.",
                [
                    f"Slice index (or elements within start, stop, step) can only be int or Tensor. ",
                    f"Got type={type(idx).__name__}.",
                ],
            )

    input_tensor = self
    if flip_dims:
        input_tensor = flip(input_tensor, dims=flip_dims)
    out = slice_helper(input_tensor, index, *args)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = squeeze(out, make_tuple(squeeze_dims))

    return out


# Conveniently converts the inputs to tensors. The decorator also fills in column info for the converted tensors.
# Because the helper is called inside another function, we need to skip one entry in the call stack to find
# the original call to user code.
@frontend_utils.convert_inputs_to_tensors(exclude=["index"], skip_num_stack_entries=1)
def slice_helper(tensor, index, *slice_params):
    return Slice.build(inputs=[tensor, *slice_params], index=index)
