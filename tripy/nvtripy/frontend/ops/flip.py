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

from typing import Optional, Sequence, Union

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int32", "int64", "bool"],
    },
)
def flip(input: "nvtripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None) -> "nvtripy.Tensor":
    r"""
    Reverses the order of elements along the specified dimension(s).

    Args:
        input: The input tensor.
        dim: The dimension(s) that should be reversed.
            If `None`, all dimensions will be reversed.

    Returns:
        A new tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(10), (2, 5))
        output = tp.flip(input) # equivalent to tp.flip(input, dim=[0, 1])
        assert cp.array_equal(cp.from_dlpack(output), cp.array([[9, 8, 7, 6, 5], [4, 3, 2, 1, 0]]))

    .. code-block:: python
        :linenos:
        :caption: Reversing only one dimension.

        input = tp.reshape(tp.arange(10), (2, 5))
        output = tp.flip(input, dim=-1)
        assert cp.array_equal(cp.from_dlpack(output), cp.array([[4, 3, 2, 1, 0], [9, 8, 7, 6, 5]]))
    """
    dim = set(op_utils.process_dim_sequence(dim, input.rank))

    slice_params = []
    for index in range(input.rank):
        if index in dim:
            slice_params.append(slice(None, None, -1))
        else:
            slice_params.append(slice(None))

    return input.__getitem__(slice_params)
