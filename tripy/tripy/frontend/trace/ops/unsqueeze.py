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

from tripy import export, constraints


@export.public_api(document_under="operations/functions")
@constraints.dtypes(
    constraints={"input": "T1", constraints.RETURN_VALUE: "T1"},
    variables={"T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"]},
)
def unsqueeze(input: "tripy.Tensor", dim: int) -> "tripy.Tensor":
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
        :caption: Example

        input = tp.iota((2, 2), dtype=tp.float32)
        output = tp.unsqueeze(input, 1)

        assert np.array_equal(cp.from_dlpack(output).get(), np.expand_dims(cp.from_dlpack(input).get(), 1))
    """
    from tripy.frontend.trace.ops.concatenate import concatenate
    from tripy.frontend.trace.ops.reshape import reshape

    from tripy.frontend import Shape

    if dim < 0:
        dim = dim + input.rank + 1

    # Add specical case for rank 0 since tensor.shape is not supported when rank is 0.
    if input.rank == 0:
        result_shape = Shape([1])
    else:
        input_shape = input.shape
        result_shape = concatenate([input_shape[:dim], Shape([1]), input_shape[dim:]], dim=0)
    return reshape(input, result_shape)
