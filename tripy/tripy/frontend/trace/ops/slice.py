#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from tripy import constraints, utils
from tripy.common.exception import raise_error
from tripy.frontend import utils as frontend_utils
from tripy.frontend.ops.registry import TENSOR_METHOD_REGISTRY
from tripy.frontend.trace.ops import utils as op_utils
from tripy.frontend.trace.ops.base import BaseTraceOp
from tripy.utils import make_tuple


@dataclass(repr=False)
class Slice(BaseTraceOp):
    shape_slice: Optional[slice] = None  # only used for inferring the length of a shape result

    def infer_dtypes(self):
        self.outputs[0].dtype = self.inputs[0].dtype

    def infer_rank(self):
        # How can we compute the output rank in the case when start, size, stride tensors are dynamic?
        self.outputs[0].rank = self.inputs[0].rank

    def infer_len(self):
        # Only infer if we have concrete values to use. Note that the result is only a Shape if these are *slices*,
        # not single indices, so a slice is the only case that needs to be considered
        if self.shape_slice is not None:
            input_len = op_utils.get_trace_shape(self.inputs[0])[0]

            def convert_to_positive_idx(idx):
                return idx if idx >= 0 else input_len + idx

            def clamp_bound(idx):
                return 0 if idx < 0 else (idx if idx <= input_len else input_len)

            stride = utils.default(self.shape_slice.step, 1)
            if stride > 0:
                start = 0 if self.shape_slice.start is None else convert_to_positive_idx(self.shape_slice.start)
                stop = input_len if self.shape_slice.stop is None else convert_to_positive_idx(self.shape_slice.stop)
            else:
                # for negative stride, we compute the indices as they would be on the flipped list, see comments below
                start = (
                    0
                    if self.shape_slice.start is None
                    else input_len - convert_to_positive_idx(self.shape_slice.start) - 1
                )
                stop = (
                    input_len
                    if self.shape_slice.stop is None
                    else input_len - convert_to_positive_idx(self.shape_slice.stop) - 1
                )

            start_point = clamp_bound(start)
            end_point = clamp_bound(stop)
            if start_point >= end_point:
                return [0]

            # - 1 because the end_point is exclusive. Use // so we round down
            strides_in_range = (end_point - start_point - 1) // abs(stride)
            # + 1 because we include the starting point and then make strides
            return [1 + strides_in_range]
        return [None]

    # we only care about the data input
    infer_shape_output_idxs = op_utils.ShapeOutputIdxPolicies.infer_from_first_input_only

    def to_flat_ir(self, inputs, outputs):
        from tripy.common.datatype import bool as tp_bool
        from tripy.common.datatype import int32
        from tripy.flat_ir.ops import DynamicReshapeOp, DynamicSliceOp
        from tripy.flat_ir.tensor import FlatIRTensor

        with FlatIRTensor.context(["construct constant tensors for slice `dim`'s > len(slice_params) // 3"]):
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
            with FlatIRTensor.context([f"generate slice index tensors for dimension {dim}"]):
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
                            shape=[1],
                            rank=1,
                            dtype=int32,
                            device=device,
                            reason_details=["reshape index tensor into singleton in case it is () instead of (1,)"],
                        )
                        shape_input = op_utils.add_constant_tensor_from_list([1], device)
                        DynamicReshapeOp.build([index_tensor, shape_input], [reshape_out])
                        return reshape_out

                    # if start > limit, the dim should be empty (we will set start to match the end)
                    def adjust_start(start_bound, end_bound):
                        from tripy.flat_ir.ops import CompareOp, SelectOp
                        from tripy.frontend.trace.ops.binary_elementwise import Comparison

                        start_comparison = FlatIRTensor.build(
                            shape=[1],
                            rank=1,
                            dtype=tp_bool,
                            device=device,
                            reason_details=["Check if start > end"],
                        )
                        adjusted_start = FlatIRTensor.build(
                            shape=[1],
                            rank=1,
                            dtype=int32,
                            device=device,
                            reason_details=["Shift the start to the end so we get an empty dimension if start > end"],
                        )

                        # pick start if it is <= end
                        CompareOp.build(
                            [start_bound, end_bound],
                            [start_comparison],
                            compare_direction=Comparison.Kind.LESS_EQUAL.compare_direction,
                        )
                        SelectOp.build([start_comparison, start_bound, end_bound], [adjusted_start])
                        return adjusted_start

                    start_bound = expand_to_rank1(slice_params[3 * dim])
                    end_bound = expand_to_rank1(slice_params[3 * dim + 1])

                    start_idxs.append(adjust_start(start_bound, end_bound))
                    limit_idxs.append(end_bound)
                    stride_idxs.append(expand_to_rank1(slice_params[3 * dim + 2]))
                else:
                    start_idxs.append(zero_1d)
                    limit_idxs.append(shape_slice)
                    stride_idxs.append(one_1d)

        with FlatIRTensor.context(["concatenate slice index tensors"]):
            start_index_tensor = op_utils.concatenate_tensors(start_idxs, dim=0)
            limit_index_tensor = op_utils.concatenate_tensors(limit_idxs, dim=0)
            stride_index_tensor = op_utils.concatenate_tensors(stride_idxs, dim=0)

        DynamicSliceOp.build([data_tensor, start_index_tensor, limit_index_tensor, stride_index_tensor], outputs)


@TENSOR_METHOD_REGISTRY("__getitem__")
@constraints.dtype_info(
    dtype_variables={
        "self_dtype": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]
    },
    dtype_constraints={"self": "self_dtype", constraints.RETURN_VALUE: "self_dtype"},
)
def __getitem__(
    self: "tripy.Tensor", index: Union[slice, int, "tripy.Tensor", Sequence[Union[slice, int, "tripy.Tensor"]]]
) -> "tripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        self: Tensor that will be sliced.
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
    from tripy.frontend.shape import Shape, ShapeScalar
    from tripy.frontend.tensor import Tensor
    from tripy.frontend.trace.ops.flip import flip
    from tripy.frontend.trace.ops.gather import gather
    from tripy.frontend.trace.ops.reshape import squeeze
    from tripy.frontend.trace.ops.where import where

    # If a tensor is indexed by another tensor, this operation is equivalent to a gather operation.
    if isinstance(index, Tensor):
        return gather(self, 0, index)

    # if we are taking a literal slice of a shape, we can pass on the slice to infer the length of the shape statically
    shape_slice = None
    if isinstance(self, Shape) and isinstance(index, slice):
        shape_slice = index

    index = make_tuple(index)
    if len(index) > self.rank:
        raise_error(f"Input tensor has a rank of {self.rank} but was attempted to be sliced with {len(index)} indices")
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
                return where(index >= 0, index, t_shape[i] + index)

        # when dealing with a slice (not a single index), we clamp the start and end bounds to [0, t_shape[i]]
        # because out of bounds indices for a *slice* mean that the dim should be empty, not an error
        def clamp_bound(bound: Union[int, Tensor]) -> Union[int, Tensor]:
            if isinstance(bound, int):
                return 0 if bound < 0 else where(bound > t_shape[i], t_shape[i], Tensor([bound]))
            else:
                return where(bound < 0, Tensor([0]), where(bound > t_shape[i], t_shape[i], bound))

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
                adjusted_start = 0 if idx.start is None else t_shape[i] - convert_to_positive_idx(idx.start) - 1
                adjusted_stop = t_shape[i] if idx.stop is None else t_shape[i] - convert_to_positive_idx(idx.stop) - 1
                args.append(clamp_bound(adjusted_start))
                args.append(clamp_bound(adjusted_stop))
            else:
                args.append(clamp_bound(convert_to_positive_idx(utils.default(idx.start, 0))))
                args.append(clamp_bound(convert_to_positive_idx(utils.default(idx.stop, t_shape[i]))))
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

    out = slice_helper(input_tensor, *args, shape_slice=shape_slice)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = squeeze(out, make_tuple(squeeze_dims))

    return ShapeScalar(out) if isinstance(self, Shape) and out.rank == 0 else out


# Conveniently converts the inputs to tensors. The decorator also fills in column info for the converted tensors.
# Because the helper is called inside another function, we need to skip one entry in the call stack to find
# the original call to user code.
@frontend_utils.convert_inputs_to_tensors(exclude=["tensor", "shape_slice"], skip_num_stack_entries=1)
def slice_helper(tensor, *slice_params, shape_slice: Optional[slice] = None):
    return Slice.build(inputs=[tensor, *slice_params], shape_slice=shape_slice)
