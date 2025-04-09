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
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.frontend.ops.binary.create import create_binary_op
from nvtripy.trace.ops.binary import LogicalOr
from nvtripy.utils import wrappers


@register_tensor_method("__or__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["bool"]},
)
def __or__(self: "nvtripy.Tensor", other: "nvtripy.Tensor") -> "nvtripy.Tensor":
    """
    Performs an elementwise logical OR.

    Args:
        self: Input tensor.
        other: The tensor to OR with this one.
            It must be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([True, False, False])
        b = tp.Tensor([False, True, False])
        output = a | b

        assert tp.equal(output, tp.Tensor([True, True, False]))
    """
    return create_binary_op(LogicalOr, self, other, cast_bool_to_int=False)
