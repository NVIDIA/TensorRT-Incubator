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
from typing import Optional, Sequence, Union

from nvtripy import export
from nvtripy.common import datatype as dt
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf
from nvtripy.frontend.ops.reduce.utils import reduce_impl
from nvtripy.trace.ops.reduce import Max
from nvtripy.frontend import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.int32, dt.int64, dt.float16, dt.bfloat16]),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def max(
    input: "nvtripy.Tensor", dim: Optional[Union[int, Sequence[int]]] = None, keepdim: bool = False
) -> "nvtripy.Tensor":
    """
    Returns a new tensor containing the maximum of the elements of the input tensor along the specified dimension.

    Args:
        input: The input tensor.
        dim: The dimension or dimensions along which to reduce.
            If this is not provided, all dimensions are reduced.
        keepdim: Whether to retain reduced dimensions in the output.
            If this is False, reduced dimensions will be squeezed.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.reshape(tp.arange(6, dtype=tp.float32), (2, 3))
        output = tp.max(input, 0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.max(np.arange(6, dtype=np.float32).reshape((2, 3)), 0))
    """
    return reduce_impl(Max, input, dim, keepdim)
