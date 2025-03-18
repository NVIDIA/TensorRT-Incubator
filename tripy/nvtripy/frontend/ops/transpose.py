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
from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
)
def transpose(input: "nvtripy.Tensor", dim0: int, dim1: int) -> "nvtripy.Tensor":
    """
    Returns a new tensor that is a transposed version of the input tensor where
    ``dim0`` and ``dim1`` are swapped.

    Args:
        input: The input tensor.
        dim0: The first dimension to be transposed.
        dim1: The second dimension to be transposed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.transpose(input, 0, 1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    from nvtripy.frontend.ops.permute import permute

    if input.rank < 2:
        raise_error("Transpose input must have at least 2 dimensions.", [f"Note: Input had {input.rank} dimensions."])

    perm = list(range(input.rank))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return permute(input, perm)
