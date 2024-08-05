#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tripy import export, dtype_info


@export.public_api(document_under="tensor_operations")
@dtype_info.dtype_info(
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
    dtype_constraints={"input": "T1", dtype_info.RETURN_VALUE: "T1"},
)
def gelu(input: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Applies an approximated Gaussian Error Linear Units (GELU) function
    to each element of the input tensor:

    :math:`\text{gelu}(x) = 0.5 * x * (1 + \tanh(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape and data type as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.gelu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert np.allclose(cp.from_dlpack(output).get(), np.from_dlpack(torch.nn.functional.gelu(t, approximate='tanh')))
    """
    from tripy.frontend.trace.ops.unary_elementwise import tanh

    t1, t2, t3, t4, t5 = 0.5, math.sqrt(2.0 / math.pi), 0.044715, 3.0, 1.0
    return t1 * input * (tanh(t2 * (input + t3 * (input**t4))) + t5)
