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

from nvtripy import export
from nvtripy.common import datatype as dt
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.frontend import wrappers
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf


@register_tensor_method("unsqueeze")
@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]
    ),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def unsqueeze(input: "nvtripy.Tensor", dim: int) -> "nvtripy.Tensor":
    """
    Returns a new tensor with the contents of the input tensor with a
    singleton dimension inserted before the specified axis.

    Args:
        input: The input tensor.
        dim: index before which to insert the singleton dimension.
            A negative dimension will be converted to ``dim = dim + input.rank + 1``.

    Returns:
        A new tensor.

    .. code-block:: python
        :linenos:

        input = tp.iota((2, 2), dtype=tp.float32)
        output = tp.unsqueeze(input, 1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.expand_dims(cp.from_dlpack(input).get(), 1))
    """
    from nvtripy.frontend.ops.reshape import reshape

    dim = op_utils.process_dim(dim, input.rank, offset=1)

    input_shape = input.shape
    result_shape = input_shape[:dim] + (1,) + input_shape[dim:]
    out = reshape(input, result_shape)

    # Since we know the shape, we can update the trace tensor accordingly.
    # This is required for some ops like conv/conv_transpose.
    out.trace_tensor.shape = input.trace_tensor.shape[:dim] + (1,) + input.trace_tensor.shape[dim:]
    return out
