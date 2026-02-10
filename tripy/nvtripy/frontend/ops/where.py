#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.where import Where
from nvtripy.types import TensorLike
from nvtripy.frontend import wrappers

from nvtripy.common import datatype as dt
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=(GetInput("condition").dtype == dt.bool)
    & OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.int8, dt.int32, dt.int64])
    & (GetInput("other").dtype == GetInput("input").dtype),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
    convert_to_tensors=True,
)
def where(condition: "nvtripy.Tensor", input: TensorLike, other: TensorLike) -> "nvtripy.Tensor":
    r"""
    Returns a new tensor of elements selected from either ``input`` or ``other``, depending on ``condition``.

    Args:
        condition: The condition tensor.
            Where this is ``True``, elements are selected from ``input``.
            Otherwise, elements are selected from ``other``.
        input: Tensor of values selected at indices where condition is ``True``.
        other: Tensor values selected at indices where condition is ``False``.

    Returns:
        A new tensor with the broadcasted shape.

    Constraints:
        All three parameters must be broadcast-compatible with each other.

    .. code-block:: python
        :linenos:

        condition = tp.Tensor([[True, False], [True, True]])
        input = tp.ones([2, 2], dtype=tp.float32)
        other = tp.zeros([2, 2], dtype=tp.float32)
        output = tp.where(condition, input, other)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[1, 0], [1, 1]], dtype=np.float32))
    """
    from nvtripy.frontend.dimension_size import DimensionSize

    condition, input, other = op_utils.match_ranks(condition, input, other)

    return op_utils.create_op(
        Where,
        [condition, input, other],
        cast_to_dimension_size=isinstance(input, DimensionSize) and isinstance(other, DimensionSize),
    )
