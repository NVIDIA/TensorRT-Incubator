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

from typing import List, Union
from tripy import export, constraints
from tripy.common.exception import raise_error


@export.public_api(document_under="operations/functions")
@constraints.dtypes(
    constraints={"tensors": "T1", constraints.RETURN_VALUE: "T1"},
    variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def stack(tensors: List[Union["tripy.Tensor"]], dim: int = 0) -> "tripy.Tensor":
    """
    Stacks multiple tensors of same shape along a given dimension.

    Args:
        tensors: List of tensors of the same shape.
        dim: The dimension to insert.

    Returns:
        A tensor with a new dimension inserted at the specified position.

    .. code-block:: python
        :linenos:
        :caption: Example

        a = tp.iota((2, 3), dtype=tp.float32)
        b = tp.iota((2, 3), dtype=tp.float32)

        output = tp.stack([a, b], dim=0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.stack((cp.from_dlpack(a).get(), cp.from_dlpack(b).get()), axis=0))
    """

    from tripy.frontend.trace.ops.unsqueeze import unsqueeze
    from tripy.frontend.trace.ops.concatenate import concatenate

    if not tensors:
        raise_error(f"tp.stack expects a non-empty list of tensors, got {tensors}")

    # Check if all tensors have the same rank
    if len(set(tensor.rank for tensor in tensors)) > 1:
        ranks = ", ".join(str(tensor.rank) for tensor in tensors)
        raise_error(f"tp.stack expects all input tensors to have the same rank, got tensors of rank {ranks}")

    expanded_tensors = [unsqueeze(tensor, dim=dim) for tensor in tensors]
    # Concatenate along the new dimension
    return concatenate(expanded_tensors, dim=dim)
