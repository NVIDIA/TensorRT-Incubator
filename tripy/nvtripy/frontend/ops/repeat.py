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

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.types import IntLike
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int32", "int64", "bool"],
    },
)
def repeat(input: "nvtripy.Tensor", repeats: IntLike, dim: int) -> "nvtripy.Tensor":
    """
    Repeats each element of a tensor after itself along the specified dimension.

    Args:
        input: The input tensor.
        repeats: The number of times to repeat each element.
        dim: The dimension along which to repeat values.

    Returns:
        The new tensor.

    .. code-block:: python
        :linenos:
        :caption: 1D tensor

        inp = tp.arange(4, dtype=tp.int32)
        out0 = tp.repeat(inp, 2, dim=0)

        cp_inp = cp.from_dlpack(inp) # doc: omit
        ref_out0 = cp.repeat(cp_inp, 2, 0) # doc: omit
        assert cp.array_equal(ref_out0, cp.from_dlpack(out0))


    .. code-block:: python
        :linenos:
        :caption: 2D tensor

        inp = tp.reshape(tp.arange(4, dtype=tp.int32), (2, 2))
        out0 = tp.repeat(inp, 2, dim=0)
        out1 = tp.repeat(inp, 2, dim=1)

        cp_inp = cp.from_dlpack(inp) # doc: omit
        ref_out0 = cp.repeat(cp_inp, 2, 0) # doc: omit
        assert cp.array_equal(ref_out0, cp.from_dlpack(out0))

        ref_out1 = cp.repeat(cp_inp, 2, 1) # doc: omit
        assert cp.array_equal(ref_out1, cp.from_dlpack(out1))
    """
    from nvtripy.frontend.dimension_size import DimensionSize
    from nvtripy.frontend.ops.expand import expand
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.unsqueeze import unsqueeze

    dim = op_utils.process_dim(dim, input.rank)

    if isinstance(repeats, int):
        if repeats < 0:
            raise_error("`repeats` value must be non-negative.", [f"Got: repeats={repeats}."])
        repeats = DimensionSize(repeats)

    # By constraining repeats to be a single integer, we can use a very
    # simple implementation for repeat.
    # Imagine we have:
    #   a = [1, 2]
    #   out = tp.repeat(a, 2, dim=0)
    #
    # We achieve this by:
    #
    # [1, 2] -> [[1],  -> [[1, 1],  -> [1, 1, 2, 2]
    #            [2],]     [2, 2],]
    #
    out = unsqueeze(input, dim + 1)
    input_shape = list(input.shape)
    out = expand(out, input_shape[: dim + 1] + [repeats] + input_shape[dim + 1 :])

    input_shape[dim] = input_shape[dim] * repeats
    return reshape(out, input_shape)
