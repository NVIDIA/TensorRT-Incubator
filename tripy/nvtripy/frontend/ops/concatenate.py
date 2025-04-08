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
from nvtripy.trace.ops.concatenate import Concatenate
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"tensors": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def concatenate(tensors: Sequence["nvtripy.Tensor"], dim: int) -> "nvtripy.Tensor":
    r"""
    Concatenates the input tensors along the specified dimension.

    Args:
        tensors: Sequence of tensors to concatenate.
                They must have identical shapes expect on the concatenation dimension.
        dim: The dimension along which the tensors are concatenated.

    Returns:
        Concatenated tensor whose shape is the same as the inputs except along ``dim``,
        whose length is the sum of the lengths of the ``dim`` axis of the inputs.

    .. code-block:: python
        :linenos:

        a = tp.iota((1, 2), dtype=tp.float32)
        b = tp.iota((2, 2), dtype=tp.float32)

        output = tp.concatenate([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.concatenate((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """
    if not tensors:
        raise_error(f"Expected a non-empty list of tensors, got {tensors}")

    if len(tensors) == 1:
        return tensors[0]

    ranks = set(tensor.rank for tensor in tensors)
    if len(ranks) > 1:
        raise_error(
            "Concatenated tensors must have equal ranks.",
            [f"Note: Input ranks were: {', '.join(str(tensor.rank) for tensor in tensors)}."],
        )

    dim = op_utils.process_dim(dim, tensors[0].rank)

    return op_utils.create_op(Concatenate, list(tensors), dim)
