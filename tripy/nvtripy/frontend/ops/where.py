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
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.where import Where
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"condition": "T2", "input": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"],
        "T2": ["bool"],
    },
)
def where(condition: "nvtripy.Tensor", input: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
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
