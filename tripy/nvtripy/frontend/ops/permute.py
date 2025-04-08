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

from typing import Sequence

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.permute import Permute
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"]},
)
def permute(input: "nvtripy.Tensor", perm: Sequence[int]) -> "nvtripy.Tensor":
    """
    Returns a tensor with its dimensions permuted.

    Args:
        input: The input tensor.
        perm: The desired ordering of dimensions.
              It must contain all integers in :math:`[0..N-1]` exactly once,
              where :math:`N` is the rank of the input tensor.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.permute(input, (1, 0))

        assert np.array_equal(cp.from_dlpack(output).get(), np.transpose(np.arange(6, dtype=np.float32).reshape(2, 3), (1, 0)))
    """
    if len(perm) != input.rank:
        raise_error(
            "Invalid permutation.",
            [
                "Permutation must have a number of elements equal to the number of dimensions in the input.\n"
                f"Note: Permutation was: {perm}, which has {len(perm)} element(s), but input has {input.rank} dimension(s)."
            ],
        )

    if list(sorted(perm)) != list(range(input.rank)):
        raise_error(
            "Invalid permutation.",
            [
                f"Permutation must contain every integer between 0 and {input.rank -1} exactly once, but permutation was: {perm}"
            ],
        )

    return op_utils.create_op(Permute, [input], perm)
