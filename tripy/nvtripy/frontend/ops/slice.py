#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy.types import IntLike
from nvtripy.utils import wrappers
from nvtripy.utils.types import type_str_from_arg
from nvtripy.utils.utils import make_list

EllipsisType = type(Ellipsis)


@register_tensor_method("__getitem__")
@wrappers.interface(
    dtype_constraints={"self": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"]},
)
def __getitem__(
    self: "nvtripy.Tensor",
    index: Union[
        "nvtripy.Tensor", slice, IntLike, EllipsisType, None, Sequence[Union[slice, IntLike, EllipsisType, None]]
    ],
) -> "nvtripy.Tensor":
    """
    Returns a tensor containing a slice of this tensor.

    Args:
        self: Tensor that will be sliced.
        index: The index or slice.
            If this is a :class:`Tensor`, the operation is equivalent to calling
            :func:`gather` along the first dimension.
            If this is `None`, a new dimension of size 1 will be inserted at that position.

    Returns:
        A tensor containing the slice of this tensor.

    .. code-block:: python
        :linenos:
        :caption: Indexing With Integers

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (3, 2))
        output = input[1]
        assert cp.array_equal(cp.from_dlpack(output), cp.from_dlpack(input)[1])

    .. code-block:: python
        :linenos:
        :caption: Indexing With Slices

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (3, 2))
        output = input[1:]
        assert cp.array_equal(cp.from_dlpack(output), cp.from_dlpack(input)[1:])

    .. code-block:: python
        :linenos:
        :caption: Indexing With Ellipsis

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (1, 3, 2))
        output = input[..., 1:]
        assert cp.array_equal(cp.from_dlpack(output), cp.from_dlpack(input)[..., 1:])

    .. code-block:: python
        :linenos:
        :caption: Reversing Data With Negative Step

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (3, 2))
        output = input[:, ::-1]
        assert cp.array_equal(cp.from_dlpack(output), cp.from_dlpack(input)[:, ::-1])

    .. code-block:: python
        :linenos:
        :caption: Indexing With Tensors (Gather)

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (3, 2))
        index = tp.Tensor([2, 0], dtype=tp.int32)
        output = input[index]
        assert cp.array_equal(cp.from_dlpack(output), cp.from_dlpack(input)[cp.array(np.from_dlpack(index))])

    .. code-block:: python
        :linenos:
        :caption: Adding New Dimensions With None

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (3, 2))
        output = input[None, :, None]
        assert output.shape == (1, 3, 1, 2)

    """
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.ops.binary.maximum import maximum
    from nvtripy.frontend.ops.binary.minimum import minimum
    from nvtripy.frontend.ops.gather import gather
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.squeeze import squeeze
    from nvtripy.frontend.ops.where import where
    from nvtripy.frontend.tensor import Tensor

    # If a tensor is indexed by another tensor, this operation is equivalent to a gather operation.
    if isinstance(index, Tensor):
        return gather(self, 0, index)

    index = [None] if index is None else make_list(index)
    ellipsis_count = index.count(Ellipsis)
    new_dim_count = index.count(None)
    if ellipsis_count > 1:
        raise_error("Slicing index can only have a single ellipsis ('...')")
    if len(index) > self.rank + new_dim_count:
        raise_error(f"Input tensor has a rank of {self.rank} but was attempted to be sliced with {len(index)} indices")
    if ellipsis_count:
        ellipsis_idx = index.index(Ellipsis)
        num_slices = self.rank + new_dim_count - len(index) + 1
        index[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * num_slices

    inp_shape = self.shape
    if new_dim_count:
        new_shape = []
        slice_indices = []
        current_dim = 0
        for idx in index:
            if idx is None:
                new_shape.append(1)
                slice_indices.append(slice(0, 1))
            else:
                new_shape.append(inp_shape[current_dim])
                slice_indices.append(idx)
                current_dim += 1

        # Add remaining dimensions
        for dim_size in inp_shape[current_dim:]:
            new_shape.append(dim_size)

        self = reshape(self, new_shape)
        inp_shape = new_shape
        index = slice_indices

    starts = []
    sizes = []
    steps = []
    squeeze_dims = []

    for dim_idx, (dim_size, slice_idx) in enumerate(zip(inp_shape, index)):

        def to_positive_idx(slice_idx):
            if isinstance(slice_idx, int):
                return slice_idx if slice_idx >= 0 else slice_idx + dim_size
            return where(slice_idx >= 0, slice_idx, dim_size + slice_idx)

        if isinstance(slice_idx, int) or isinstance(slice_idx, DimensionSize):
            slice_idx = to_positive_idx(slice_idx)
            starts.append(slice_idx)
            sizes.append(1)
            steps.append(1)

            squeeze_dims.append(dim_idx)
        else:
            assert isinstance(slice_idx, slice)

            def check_type(name, arg):
                if arg is not None and not isinstance(arg, (int, DimensionSize)):
                    raise_error(
                        f"Slice {name} must be an integer or a DimensionSize.",
                        [f"Note: At index: {dim_idx}, {name} was of type '{type_str_from_arg(arg)}': ", arg],
                    )

            check_type("start", slice_idx.start)
            check_type("stop", slice_idx.stop)
            check_type("step", slice_idx.step)

            step = utils.utils.default(slice_idx.step, 1)

            def cast_to_dim_size(arg):
                if not isinstance(arg, DimensionSize):
                    arg = DimensionSize(arg)
                return arg

            # Ternary select that can accept `cond` as either a DimensionSize or bool
            def select(cond, lhs, rhs):
                if isinstance(cond, bool):
                    return lhs if cond else rhs
                return where(cond, cast_to_dim_size(lhs), cast_to_dim_size(rhs))

            # For negative step sizes, the default start/stop are inverted.
            default_start = select(step >= 0, 0, dim_size - 1)
            default_stop = select(step >= 0, dim_size, -1)

            def get_min(a, b):
                return (
                    min(a, b)
                    if isinstance(a, int) and isinstance(b, int)
                    else minimum(cast_to_dim_size(a), cast_to_dim_size(b))
                )

            if slice_idx.start is not None:
                start = to_positive_idx(slice_idx.start)
                # If `start` is past the end, clamp it - if we're going backwards, we need to clamp it to a valid value;
                # otherwise, we can clamp it out of bounds (which will yield an empty tensor):
                start = get_min(start, select(step >= 0, dim_size, dim_size - 1))
            else:
                start = default_start

            if slice_idx.stop is not None:
                stop = to_positive_idx(slice_idx.stop)
                # Must clamp `stop` so that slicing past the end behaves as expected.
                stop = get_min(stop, dim_size)
            else:
                stop = default_stop

            # Need to convert `stop` to a `size`:
            size = stop - start
            if not op_utils.is_int_equal_to(step, 1):
                size = op_utils.int_ceil_div(size, step)

            # Size cannot be less than 0:
            size = max(size, 0) if isinstance(size, int) else maximum(size, cast_to_dim_size(0))

            starts.append(start)
            sizes.append(size)
            steps.append(step)

    # For any dimensions omitted in `index`, include the full extent of the input dimension
    for dim_size in inp_shape[len(index) :]:
        starts.append(0)
        sizes.append(dim_size)
        steps.append(1)

    starts = op_utils.tensor_from_shape_like(starts)
    sizes = op_utils.tensor_from_shape_like(sizes)
    steps = op_utils.tensor_from_shape_like(steps)

    slice_out = op_utils.create_op(Slice, [self, starts, sizes, steps])

    return squeeze(slice_out, squeeze_dims)
