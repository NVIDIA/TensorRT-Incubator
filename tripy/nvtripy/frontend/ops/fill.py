#
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
#

from typing import Optional

from nvtripy import export, utils
from nvtripy.common import datatype
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.fill import Fill
from nvtripy.types import ShapeLike, TensorLike
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
)
def full(shape: ShapeLike, value: TensorLike, dtype: "nvtripy.dtype" = datatype.float32) -> "nvtripy.Tensor":
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
    return op_utils.create_op(Fill, [shape, value], dtype=dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"input": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
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
    return op_utils.create_op(
        Fill, [op_utils.tensor_from_shape_like(input.shape), value], dtype=utils.utils.default(dtype, input.dtype)
    )
