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
from tripy import constraints, export

from tripy.frontend import utils as frontend_utils


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int32"],
    },
    dtype_constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
)
@frontend_utils.process_dim
def cumsum(input: "tripy.Tensor", dim: int) -> "tripy.Tensor":
    """
    Computes the cumulative sum of elements in the input along the dimension ``dim``.

    Args:
        input: The input tensor.
        dim: The dimension along which to compute the cumulative sum.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:
        :caption: 1D tensor

        input = tp.arange(4, 0, step=-1, dtype=tp.int32)
        output = tp.cumsum(input, dim=0)

        assert cp.array_equal(cp.cumsum(cp.from_dlpack(input)), cp.from_dlpack(output))

    .. code-block:: python
        :linenos:
        :caption: 2D tensor

        input = tp.reshape(tp.arange(9, 0, step=-1, dtype=tp.int32), (3, 3))
        output = tp.cumsum(input, dim=0)

        assert cp.array_equal(cp.cumsum(cp.from_dlpack(input), axis=0), cp.from_dlpack(output))
    """
    # Consider:
    #
    #   a = [3, 2, 1]
    #
    # then, we can implement cumsum as:
    #
    #  out = a @ [[1, 1, 1]
    #             [0, 1, 1]
    #             [0, 0, 1]]
    #
    # which will yield:
    #
    #   out = [3, 3 + 2, 3 + 2 + 1]
    #
    # In the general case where `a` is an N-dimensional tensor, we simply transpose
    # the dimension of interest to the innermost position and then carry out the
    # GEMM described above, then tranpose the output back.
    from tripy.frontend.trace.ops.permute import permute
    from tripy.frontend.ops.tensor_initializers import triu, ones

    # For the examples in the comments that follow, assume the input shape is (3, 5, 7) and
    # we are applying cumsum over dim=1 (the dimension of length 5).

    # Swap dim to innermost position: (3, 5, 7) -> (3, 7, 5)
    move_to_innermost_perm = list(range(input.rank))
    del move_to_innermost_perm[dim]
    move_to_innermost_perm.append(dim)
    transposed = permute(input, move_to_innermost_perm)

    # GEMM with square upper triangular matrix: (3, 7, 5) @ (5, 5) -> (3, 7, 5)

    # TODO: We should replace this with:
    #   shape = transposed.shape[-1:] * 2
    # once the relevant shape inference bugs are fixed.
    shape = (transposed.shape[input.rank - 1], transposed.shape[input.rank - 1])
    out = transposed @ triu(ones(shape=shape, dtype=transposed.dtype))

    # Swap innermost position back to `dim`: (3, 7, 5) -> (3, 5, 7)
    reset_dim_perm = list(range(input.rank))
    del reset_dim_perm[-1]
    reset_dim_perm.insert(dim, input.rank - 1)
    out = permute(out, reset_dim_perm)

    return out
