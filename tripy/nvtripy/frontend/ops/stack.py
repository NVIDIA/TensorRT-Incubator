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

from typing import Sequence

from nvtripy import export
from nvtripy.common.exception import raise_error
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"tensors": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int4", "int8", "int32", "int64", "bool"],
    },
)
def stack(tensors: Sequence["nvtripy.Tensor"], dim: int = 0) -> "nvtripy.Tensor":
    """
    Stacks multiple tensors of same shape along a given dimension.

    Args:
        tensors: Sequence of tensors of the same shape.
        dim: The dimension to insert.

    Returns:
        A tensor with a new dimension inserted at the specified position.

    .. code-block:: python
        :linenos:

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((2, 3), dtype=tp.float32)

        output = tp.stack([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.stack((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """

    from nvtripy.frontend.ops.unsqueeze import unsqueeze
    from nvtripy.frontend.ops.concatenate import concatenate

    if not tensors:
        raise_error(f"Expected a non-empty list of tensors, got {tensors}")

    # Check if all tensors have the same rank
    if len(set(tensor.rank for tensor in tensors)) > 1:
        ranks = ", ".join(str(tensor.rank) for tensor in tensors)
        raise_error(
            f"Expected all input tensors to have the same rank.", [f"Note: Got tensors of multiple ranks: {ranks}."]
        )

    expanded_tensors = [unsqueeze(tensor, dim=dim) for tensor in tensors]
    # Concatenate along the new dimension
    return concatenate(expanded_tensors, dim=dim)
