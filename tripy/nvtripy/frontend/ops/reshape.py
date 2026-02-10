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

import math

from nvtripy import export
from nvtripy.common import datatype as dt
from nvtripy.common.exception import raise_error
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.reshape import Reshape
from nvtripy.types import ShapeLike
from nvtripy.frontend import wrappers
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf


def infer_dimensions(input: "nvtripy.Tensor", shape: ShapeLike) -> ShapeLike:

    num_unknown_dims = len([dim for dim in shape if op_utils.is_int_equal_to(dim, -1)])
    if num_unknown_dims > 1:
        raise_error(f"The new shape can have at most one inferred dimension (denoted by -1)", [f"Got shape: {shape}."])

    if num_unknown_dims == 1:
        input_volume = math.prod(input.shape)
        known_dims_volume = math.prod(dim for dim in shape if not op_utils.is_int_equal_to(dim, -1))
        # If we have scalars, the floor div ensures the result is an int:
        inferred_dim = input_volume // known_dims_volume

        shape = [inferred_dim if op_utils.is_int_equal_to(dim, -1) else dim for dim in shape]

    return {"shape": shape}


@register_tensor_method("reshape")
@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.int4, dt.int8, dt.int32, dt.int64, dt.bool]
    ),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
    convert_to_tensors=True,
    conversion_preprocess_func=infer_dimensions,
)
def reshape(input: "nvtripy.Tensor", shape: ShapeLike) -> "nvtripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor in the specified shape.

    Args:
        input: The input tensor.
        shape: The desired compatible shape. If a shape dimension is -1, its value
            is inferred based on the other dimensions and the number of elements in the input.
            Atmost one dimension can be -1.

    Returns:
        A new tensor with the specified shape.

    .. code-block:: python
        :linenos:

        input = tp.iota((2, 3), dtype=tp.float32)
        output = tp.reshape(input, (1, 6))

        assert np.array_equal(cp.from_dlpack(output).get(), np.reshape(cp.from_dlpack(input).get(), (1, 6)))
    """
    return op_utils.create_op(Reshape, [input, shape])
