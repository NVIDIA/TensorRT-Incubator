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
import math

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"]},
)
def flatten(input: "nvtripy.Tensor", start_dim: int = 0, end_dim: int = -1) -> "nvtripy.Tensor":
    """
    Flattens the input tensor from start_dim to end_dim.

    Args:
        input: The input tensor to be flattened.
        start_dim: The first dimension to flatten (default is 0).
        end_dim: The last dimension to flatten (default is -1, which includes the last dimension).

    Returns:
        A flattened tensor.

    .. code-block:: python
        :linenos:
        :caption: Flatten All Dimensions

        input = tp.iota((1, 2, 1), dtype=tp.float32)
        output = tp.flatten(input)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().flatten())

    .. code-block:: python
        :linenos:
        :caption: Flatten Starting from First Dimension

        input = tp.iota((2, 3, 4), dtype=tp.float32)
        output = tp.flatten(input, start_dim=1)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().reshape(2, -1))

    .. code-block:: python
        :linenos:
        :caption: Flatten a Specific Range of Dimensions

        input = tp.iota((2, 3, 4, 5), dtype=tp.float32)
        output = tp.flatten(input, start_dim=1, end_dim=2)
        assert np.array_equal(cp.from_dlpack(output).get(), cp.from_dlpack(input).get().reshape(2, -1, 5))
    """
    from nvtripy.frontend.ops.reshape import reshape

    start_dim = op_utils.process_dim(start_dim, input.rank)
    end_dim = op_utils.process_dim(end_dim, input.rank)

    # Ensure start_dim and end_dim are within the valid range.
    if start_dim > end_dim:
        raise_error(
            f"`start_dim` cannot be larger than `end_dim`.", [f"Note: start_dim={start_dim}, end_dim={end_dim}."]
        )

    shape = input.shape
    flattened_dim_size = math.prod(shape[start_dim : end_dim + 1])
    flattened_shape = shape[:start_dim] + (flattened_dim_size,) + shape[end_dim + 1 :]
    return reshape(input, flattened_shape)
