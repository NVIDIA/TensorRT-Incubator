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
from nvtripy.common import datatype
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.linspace import Linspace
from nvtripy.types import ShapeLike
from nvtripy.utils import wrappers


def iota_impl(shape: "nvtripy.Tensor", dim: int, dtype: datatype.dtype) -> "nvtripy.Tensor":
    from nvtripy.frontend.ops.cast import cast
    from nvtripy.frontend.tensor import Tensor

    linspace_dtype = Linspace.get_closest_dtype(dtype)
    start = Tensor(0, dtype=linspace_dtype)

    step = [0] * op_utils.get_shape_len(shape)  # output rank
    step[dim] = 1
    step = Tensor(step, dtype=linspace_dtype)

    out = op_utils.create_op(Linspace, [shape, start, step], dtype=linspace_dtype)
    return cast(out, dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"dtype": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
    convert_to_tensors=True,
)
def iota(shape: ShapeLike, dim: int = 0, dtype: datatype.dtype = datatype.float32) -> "nvtripy.Tensor":
    """
    Fills an output tensor with consecutive values starting from zero along the given dimension.

    Args:
        shape: The desired shape.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type.

    Returns:
        A tensor of shape ``shape`` and data type ``dtype``.

    .. code-block:: python
        :linenos:

        output = tp.iota((3,), dim=-1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    dim = op_utils.process_dim(dim, op_utils.get_shape_len(shape))

    return iota_impl(shape, dim, dtype)


@export.public_api(document_under="operations/initializers")
@wrappers.interface(
    dtype_constraints={"input": "T1", "dtype": "T2", wrappers.RETURN_VALUE: "T2"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
    },
)
def iota_like(input: "nvtripy.Tensor", dim: int = 0, dtype: Optional[datatype.dtype] = None) -> "nvtripy.Tensor":
    """
    Returns a tensor of the same shape and data type as the input tensor, with consecutive values
    starting from zero along the given dimension.

    Args:
        input: Input tensor.
        dim: Dimension along which to perform the iota operation.
            This cannot exceed the rank of the specified shape.
        dtype: The desired data type. This will override the data type inferred from the input tensor.

    Returns:
        A tensor of the same shape and data type (unless ``dtype`` is provided) as the input.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([1, 2, 3])
        output = tp.iota_like(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.arange(0, 3, dtype=np.float32))
    """
    dim = op_utils.process_dim(dim, input.rank)

    return iota_impl(
        op_utils.tensor_from_shape_like(input.shape),
        dim,
        utils.utils.default(dtype, input.dtype),
    )
