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
from typing import Tuple

from nvtripy import export
from nvtripy.frontend.ops.reduce.utils import topk_impl
from nvtripy.trace.ops.topk import TopKMax
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: ["T1", "T2"]},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "int64"], "T2": ["int32"]},
)
def topk(input: "nvtripy.Tensor", k: int, dim: int) -> Tuple["nvtripy.Tensor", "nvtripy.Tensor"]:
    """
    Returns the ``k`` largest values in the tensor and their
    indices along the specified dimension.

    Args:
        input: The input tensor.
        k: The number of values to take.
        dim: The dimension along which to find the top-k values.

    Returns:
        The top-k values and indices, in sorted order.

    .. code-block:: python
        :linenos:

        inp = tp.iota((1, 5), dim=1) + 2.5
        values, indices = tp.topk(inp, k=2, dim=1)

        assert tp.equal(values, tp.Tensor([[6.5, 5.5]]))
        assert tp.equal(indices, tp.Tensor([[4, 3]]))
    """
    return topk_impl(TopKMax, input, k=k, dim=dim)
