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
from nvtripy import export
from nvtripy.common import datatype as dt
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf
from nvtripy.frontend.ops.binary.create import create_binary_op
from nvtripy.trace.ops.binary import Max
from nvtripy.types import TensorLike
from nvtripy.frontend import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(
        GetInput("lhs").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64, dt.bool]
    )
    & (GetInput("rhs").dtype == GetInput("lhs").dtype),
    output_guarantees=GetReturn(0).dtype == GetInput("lhs").dtype,
    convert_to_tensors=True,
)
def maximum(lhs: TensorLike, rhs: TensorLike) -> "nvtripy.Tensor":
    """
    Performs an elementwise maximum.

    Args:
        lhs: The first input tensor.
        rhs: The second input tensor.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([1.0, 6.0])
        b = tp.Tensor([2.0, 3.0])
        output = tp.maximum(a, b)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 6.0]))
    """
    return create_binary_op(Max, lhs, rhs)
