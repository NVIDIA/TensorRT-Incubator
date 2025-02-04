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

import numbers

from nvtripy import export
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.trace.ops.where import Where
from nvtripy.utils import wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"condition": "T2", "input": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
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
    return op_utils.create_op(Where, [condition, input, other])


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", "mask": "T2", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16", "float8", "int4", "int8", "int32", "int64", "bool"],
        "T2": ["bool"],
    },
)
def masked_fill(input: "nvtripy.Tensor", mask: "nvtripy.Tensor", value: numbers.Number) -> "nvtripy.Tensor":
    r"""
    Returns a new tensor filled with ``value`` where ``mask`` is ``True`` and elements from
    the input tensor otherwise.

    Args:
        input: The input tensor.
        mask: The mask tensor.
        value: the value to fill with. This will be casted to match the data type of the input tensor.

    Returns:
        A new tensor of the same shape as the input tensor.

    .. code-block:: python
        :linenos:

        mask = tp.Tensor([[True, False], [True, True]])
        input = tp.zeros([2, 2])
        output = tp.masked_fill(input, mask, -1.0)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([[-1, 0], [-1, -1]], dtype=np.float32))
    """
    from nvtripy.frontend.ops.fill import full_like

    fill_tensor = full_like(input, value)
    return where(mask, fill_tensor, input)
