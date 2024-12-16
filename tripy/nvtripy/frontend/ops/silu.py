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

from nvtripy import export, wrappers


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    dtype_constraints={"input": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={
        "T1": ["float32", "float16", "bfloat16"],
    },
)
def silu(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function  to each element of the
    input tensor. This function is also known as the swish function.

    :math:`\text{silu}(x) = x \cdot \sigma (x)`
    where
    :math:`\sigma (x)_i = \frac{1}{1 + \exp{-x_i}}`

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.Tensor([1., 2., 3., 4.], dtype=tp.float32)
        output = tp.silu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch.nn.functional.silu(t)))
    """
    from nvtripy.frontend.trace.ops.unary_elementwise import exp

    return input / (1.0 + exp(-1.0 * input))
