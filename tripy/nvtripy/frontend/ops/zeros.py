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
from nvtripy.common import datatype
from nvtripy.frontend.ops.full import full, full_like
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"],
    },
)
def zeros(
    shape: "nvtripy.types.ShapeLike",
    dtype: datatype.dtype = datatype.float32,
) -> "nvtripy.Tensor":
    """
    Creates a Tensor of the specified shape and dtype with all elements set to 0.

    Args:
        shape: The desired shape of the tensor.
        dtype: Datatype of elements.

    Returns:
        A tensor of shape ``shape`` with all elements set to 0.

    .. code-block:: python
        :linenos:

        output = tp.zeros([2, 3])

        assert np.array_equal(cp.from_dlpack(output).get(), np.zeros([2, 3], dtype=np.float32))

    .. seealso:: :func:`zeros_like`, :func:`full`
    """
    return full(shape, 0.0, dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"input": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"],
    },
)
def zeros_like(input: "nvtripy.Tensor", dtype: Optional[datatype.dtype] = None) -> "nvtripy.Tensor":
    """
    Creates a Tensor with all elements set to 0 of the same shape as the input tensor.

    Args:
        input: The input tensor.
        dtype: Datatype of elements. If set to ``None``, the datatype of the input tensor is used.

    Returns:
        A tensor of the same shape as the input with all elements set to 0.

    .. code-block:: python
        :linenos:

        input = tp.iota([2, 3], dtype=tp.float32)
        output = tp.zeros_like(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.zeros([2, 3], dtype=np.float32))

    .. seealso:: :func:`zeros`, :func:`full_like`
    """

    return full_like(input, 0.0, dtype)
