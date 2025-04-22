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

from typing import Sequence, Tuple, Union

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.types import IntLike
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"],
    },
)
def split(
    input: "nvtripy.Tensor", num_split_or_sizes: Union[int, Sequence[IntLike]], dim: int = 0
) -> Tuple["nvtripy.Tensor"]:
    r"""
    Splits a tensor along the specified dimension.

    Args:
        input: The input tensor.

        num_split_or_sizes:
            If this is an integer, the input is split into this many equal sized chunks.
            If the dimension cannot be divided evenly, the last chunk will be smaller.

            If this is a sequence of values, the input will be split into ``len(num_split_or_sizes)``
            chunks where the :math:`i^{th}` chunk has a size of ``num_split_or_sizes[i]``.
            The size of the chunk will be clamped if the input is too small.

        dim: The dimension along which the slices are done. All other dimensions are included in full.

    Returns:
        A tuple of slices of the input tensor.

    .. code-block:: python
        :linenos:
        :caption: Splitting Into 2 Chunks

        # doc: print-locals input outputs
        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[2:, :]).get())

    .. code-block:: python
        :linenos:
        :caption: Splitting Along A Different Dimension

        # doc: print-locals input outputs
        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, 2, dim=1)
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:, :2]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[:, 2:]).get())

    .. code-block:: python
        :linenos:
        :caption: Splitting With Custom Chunk Sizes

        # doc: print-locals input outputs
        input = tp.reshape(tp.arange(16, dtype=tp.float32), (4, 4))
        outputs = tp.split(input, [1, 1, 2])
        assert np.array_equal(cp.from_dlpack(outputs[0]).get(), cp.from_dlpack(input[:1, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[1]).get(), cp.from_dlpack(input[1:2, :]).get())
        assert np.array_equal(cp.from_dlpack(outputs[2]).get(), cp.from_dlpack(input[2:, :]).get())
    """
    dim = op_utils.process_dim(dim, input.rank)

    if isinstance(num_split_or_sizes, int):
        if num_split_or_sizes <= 0:
            raise_error(f"`num_split_or_sizes` must be positive, but got: {num_split_or_sizes}")

        chunk_sizes = [op_utils.int_ceil_div(input.shape[dim], num_split_or_sizes)] * num_split_or_sizes
    else:
        if not num_split_or_sizes:
            raise_error("Split indices must not be empty")
        chunk_sizes = num_split_or_sizes

    def slice_on_dim(start, stop):
        slice_params = []
        for index in range(input.rank):
            if index == dim:
                slice_params.append(slice(start, stop))
            else:
                slice_params.append(slice(None))
        return input.__getitem__(slice_params)

    splits = []
    start = 0
    for size in chunk_sizes:
        splits.append(slice_on_dim(start, start + size))
        start += size

    return tuple(splits)
