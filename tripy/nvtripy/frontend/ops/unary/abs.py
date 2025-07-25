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


from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.ops._registry import register_tensor_method
from nvtripy.trace.ops.unary import Abs
from nvtripy.utils import wrappers


@register_tensor_method("__abs__")
@wrappers.interface(
    dtype_constraints={"self": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "int8", "int32", "int64"]},
)
def __abs__(self: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Computes the elementwise absolute value of the input tensor.

    Args:
        self: The input tensor.

    Returns:
        A new tensor of the same shape.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([-1, -2])
        output = abs(input)

        assert np.array_equal(cp.from_dlpack(output).get(), np.array([1, 2], dtype=np.float32))
    """
    return op_utils.create_op(Abs, [self])
