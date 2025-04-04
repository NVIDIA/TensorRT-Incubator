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

from typing import Sequence, Union

from nvtripy import utils
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.slice import Slice
from nvtripy.types import TensorLike
from nvtripy.utils import wrappers
from nvtripy.utils.utils import make_tuple


@register_tensor_method("__getitem__")
@wrappers.interface(
    dtype_constraints={"self": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
)
def __getitem__(
    self: "nvtripy.Tensor", index: Union[slice, int, "nvtripy.Tensor", Sequence[Union[slice, int, "nvtripy.Tensor"]]]
) -> "nvtripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        self: Tensor that will be sliced.
        index: The index (as an int or Tripy tensor) or slice.

    Returns:
        A tensor containing the slice of this tensor.

    .. code-block:: python
        :linenos:

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
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.ops.binary_elementwise import maximum, minimum
    from nvtripy.frontend.ops.flip import flip
    from nvtripy.frontend.ops.gather import gather
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.where import where
    from nvtripy.frontend.tensor import Tensor

    # If a tensor is indexed by another tensor, this operation is equivalent to a gather operation.
    if isinstance(index, Tensor):
        return gather(self, 0, index)

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
                if bound < 0:
                    return 0

                if isinstance(t_shape[i], int):
                    return min(bound, t_shape[i])
                return minimum(t_shape[i], Tensor([bound]))

            # need the shame dimension to be a tensor to use as an argument to min and max
            shape_dim = t_shape[i] if isinstance(t_shape[i], Tensor) else DimensionSize(t_shape[i])
            return maximum(Tensor([0]), minimum(shape_dim, bound))

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
                args.append(clamp_bound(convert_to_positive_idx(utils.utils.default(idx.start, 0))))
                args.append(clamp_bound(convert_to_positive_idx(utils.utils.default(idx.stop, t_shape[i]))))
            args.append(abs(utils.utils.default(idx.step, 1)))
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

    out = slice_helper(input_tensor, *args)

    squeeze_dims = []
    for i, idx in enumerate(index):
        if isinstance(idx, (tuple, list)):
            raise NotImplementedError("Gather is not supported")
        if isinstance(idx, int):
            squeeze_dims.append(i)
    if squeeze_dims:
        out = squeeze(out, make_tuple(squeeze_dims))

    return out


@wrappers.interface(convert_to_tensors=True)
def slice_helper(tensor, *slice_params: TensorLike):
    from nvtripy.utils import function_registry
    from nvtripy.utils.ast import get_arg_candidate_column_offsets

    # The default behavior of the tensor conversion will not add the correct column info to the slice params
    # because this call occurs *inside* the overridden call to __getitem__, so we adjust the column info manually.
    # Look for the stack frame index to __getitem__. We need to go one stack frame beyond to get to the *user* call of __getitem__.
    def find_frame_index(arg):
        # Internal WAR: the constraints decorator is applied before the function registry decorator, so in the constraints tests,
        # we will not find the Tensor.__getitem__ decorator. We can use a fallback in that case.
        frame_index = -1
        function_registry_wrapper_found = False
        for idx, source_info in enumerate(arg.stack_info):
            if source_info.module == function_registry.__name__ and source_info.function == "wrapper":
                function_registry_wrapper_found = True
            if source_info._dispatch_target == "Tensor.__getitem__":
                frame_index = idx + 1
                break

        assert (
            not function_registry_wrapper_found or frame_index >= 0
        ), "No call to the Tensor.__getitem__ dispatch found"
        return frame_index if function_registry_wrapper_found else arg.stack_info.get_first_user_frame_index()

    assert slice_params

    # The frame index will *usually* be the same across params, but in some cases (when clamping bounds for slices),
    # the step parameters might have shorter stack depths.
    frame_index = find_frame_index(slice_params[0])

    arg_names = ["tensor"] + ["slice_params"] * len(slice_params)
    for arg_index, arg in enumerate(slice_params):
        arg_frame_index = frame_index
        if (
            arg_frame_index > len(arg.stack_info)
            or arg.stack_info[arg_frame_index]._dispatch_target != "Tensor.__getitem__"
        ):
            arg_frame_index = find_frame_index(arg)

        source_info = arg.stack_info[arg_frame_index]

        # Note: arg_index does not account for the positional arg, hence we add 1 for the index argument
        # Also, strip the "Tensor" prefix from the dispatch target.
        candidates = get_arg_candidate_column_offsets(
            source_info.code, 1 + arg_index, 1, "__getitem__", False, arg_names
        )

        # Now we can set the column range correctly
        if len(candidates) == 1:
            source_info.column_range = candidates[0]

    return op_utils.create_op(Slice, inputs=[tensor, *slice_params])
