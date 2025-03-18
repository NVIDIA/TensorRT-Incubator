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
from nvtripy.types import TensorLike
from nvtripy.utils import wrappers


def mod_impl(lhs, rhs):
    return lhs - ((lhs // rhs) * rhs)


@register_tensor_method("__mod__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    convert_to_tensors=True,
)
def __mod__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a modulo operation, which computes the remainder of a division.

    Args:
        self: Input tensor.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        b = tp.Tensor([3.0, 4.0])
        output = a % b

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1.0, 2.0]))
    """
    return mod_impl(self, other)


@register_tensor_method("__rmod__")
@wrappers.interface(
    dtype_constraints={"self": "T1", "other": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64", "bool"]},
    convert_to_tensors=True,
)
def __rmod__(self: "nvtripy.Tensor", other: TensorLike) -> "nvtripy.Tensor":
    """
    Performs a modulo operation, which computes the remainder of a division.

    Args:
        self: Input tensor.
        other: The tensor by which to divide `self`.
            It should be broadcast-compatible.

    Returns:
        A new tensor with the broadcasted shape.

    .. code-block:: python
        :linenos:

        a = tp.Tensor([4.0, 6.0])
        output = 2 % a

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([2.0, 2.0]))
    """
    return mod_impl(other, self)
