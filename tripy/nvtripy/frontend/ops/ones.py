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
from nvtripy.common import datatype as dt
from nvtripy.frontend import wrappers
from nvtripy.frontend.constraints import GetInput, GetReturn, If, OneOf
from nvtripy.frontend.ops.full import full, full_like


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]
    ),
    output_guarantees=GetReturn(0).dtype == GetInput("dtype"),
)
def ones(
    shape: "nvtripy.types.ShapeLike",
    dtype: dt.dtype = dt.float32,
) -> "nvtripy.Tensor":
    """
    Creates a Tensor of the specified shape and dtype with all elements set to 1.

    Args:
        shape: The desired shape of the tensor.
        dtype: Datatype of elements.

    Returns:
        A tensor of shape ``shape`` with all elements set to 1.

    .. code-block:: python
        :linenos:

        output = tp.ones([2, 3])

        assert np.array_equal(cp.from_dlpack(output).get(), np.ones([2, 3], dtype=np.float32))

    .. seealso:: :func:`ones_like`, :func:`full`
    """
    return full(shape, 1.0, dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("input").dtype,
        [dt.float32, dt.float16, dt.bfloat16, dt.float8, dt.int8, dt.int32, dt.int64, dt.bool],
    )
    & If(
        GetInput("dtype") != None,
        OneOf(
            GetInput("dtype"),
            [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool],
        ),
    ),
    output_guarantees=If(
        GetInput("dtype") != None,
        GetReturn(0).dtype == GetInput("dtype"),
        GetReturn(0).dtype == GetInput("input").dtype,
    ),
)
def ones_like(input: "nvtripy.Tensor", dtype: Optional[dt.dtype] = None) -> "nvtripy.Tensor":
    """
    Creates a tensor with all elements set to 1 of the same shape as the input tensor.

    Args:
        input: The input tensor.
        dtype: Datatype of elements. If set to ``None``, the datatype of the input tensor is used.

    Returns:
        A tensor of the same shape as the input with all elements set to 1.

    .. code-block:: python
        :linenos:

        input = tp.zeros([2, 3], dtype=tp.float32)
        output = tp.ones_like(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.ones([2, 3], dtype=np.float32))

    .. seealso:: :func:`ones`, :func:`full_like`
    """
    return full_like(input, 1.0, dtype)
