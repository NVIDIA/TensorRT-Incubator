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
from typing import Optional

from nvtripy import export
from nvtripy.frontend.ops.reduce.utils import arg_min_max_impl
from nvtripy.trace.ops.topk import TopKMax
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int32", "int64"], "T2": ["int32"]},
)
def argmax(input: "nvtripy.Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "nvtripy.Tensor":
    """
    Returns a new tensor containing the indices of maximum values
    of the input tensor along the specified dimension.

    If there are multiple maximum values, then the indices of the
    first maximum value are returned.

    Args:
        input: The input tensor.
        dim: The dimension along which to reduce.
            If this is not provided, the index of the flattened input is returned.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([[1.0, 0.0, 3.0], [0.5, 2.0, 1.5]])
        output = tp.argmax(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.argmax([[1.0, 0.0, 3.0], [0.5, 2.0, 1.5]], 0))
    """
    return arg_min_max_impl(TopKMax, input, dim, keepdim)
