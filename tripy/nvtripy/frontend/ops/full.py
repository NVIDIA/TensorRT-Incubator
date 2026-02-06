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

from typing import Optional

from nvtripy import export, utils
from nvtripy.common import datatype as dt
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf, If
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.broadcast import Broadcast
from nvtripy.types import ShapeLike, TensorLike
from nvtripy.frontend import wrappers


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    input_requirements=(GetInput("value").dtype != dt.float8)
    & If(
        GetInput("value").dtype == dt.int8,
        GetInput("dtype") != dt.bool,
    )
    & OneOf(GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]),
    output_guarantees=GetReturn(0).dtype == GetInput("dtype"),
    convert_to_tensors=True,
)
def full(shape: ShapeLike, value: TensorLike, dtype: "nvtripy.dtype" = dt.float32) -> "nvtripy.Tensor":
    """
    Returns a tensor of the desired shape with all values set to the specified value.

    Args:
        shape: The desired shape.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape``.

    .. code-block:: python
        :linenos:

        output = tp.full(shape=[2, 3], value=2)

        assert np.array_equal(cp.from_dlpack(output).get(), np.full([2, 3], 2, dtype=np.float32))
    """
    from nvtripy.frontend.ops.cast import cast

    value_dtype = dtype
    if dtype == dt.int8:
        # TODO (#580): Remove this workaround for broadcasting INT8 inputs:
        value_dtype = dt.int32

    # We avoid using the `expand` API since it does extra things that we don't need.
    out = op_utils.create_op(Broadcast, [cast(value, dtype=value_dtype), shape])
    out = cast(out, dtype=dtype)  # This will be a no-op if dtype == value_dtype
    return out


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.float8, dt.int8, dt.int32, dt.int64, dt.bool]
    )
    & (GetInput("value").dtype != dt.float8)
    & If(
        GetInput("value").dtype == dt.int8,
        If(
            GetInput("dtype") != None,
            GetInput("dtype") != dt.bool,
            GetInput("input").dtype != dt.bool,
        ),
    )
    & If(
        GetInput("dtype") != None,
        OneOf(GetInput("dtype"), [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]),
    ),
    output_guarantees=If(
        GetInput("dtype") != None,
        GetReturn(0).dtype == GetInput("dtype"),
        GetReturn(0).dtype == GetInput("input").dtype,
    ),
)
def full_like(input: "nvtripy.Tensor", value: TensorLike, dtype: Optional["nvtripy.dtype"] = None) -> "nvtripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with all values
    set to the specified value.

    Args:
        input: Input tensor.
        value: A scalar value to fill the resulting tensor.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([[1, 2], [3, 4]])
        output = tp.full_like(input, value=2)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[2, 2], [2, 2]], dtype=np.float32))
    """
    return full(input.shape, value, utils.utils.default(dtype, input.dtype))
