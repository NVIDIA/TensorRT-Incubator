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
def softmax(input: "nvtripy.Tensor", dim: int = None) -> "nvtripy.Tensor":
    r"""
    Applies the softmax function to the input tensor:

    :math:`\text{softmax}(x_{i}) = \Large \frac{e^{x_{i}}}{\sum_{j=1}^N e^{x_{j}}} \normalsize for\ i=1,2,\dots,N`

    where :math:`x_{i}` is the :math:`i^{th}` element along dimension ``dim``
    and :math:`N` is the size of the dimension.

    Effectively, for each slice along ``dim``, elements are scaled such that they
    lie in the range :math:`[0, 1]` and sum to 1.

    Args:
        input: The input tensor.
        dim: The dimension along which softmax will be computed.
            If this is ``None``, softmax is applied over the whole input array.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:
        :caption: Example

        input = tp.iota([2, 2], dtype=tp.float32)
        output = tp.softmax(input, dim=0)

        assert tp.allclose(output, tp.Tensor(torch.Tensor([[0., 0.], [1., 1.]]).softmax(0)))
    """
    from nvtripy.frontend.trace.ops.reduce import max, sum
    from nvtripy.frontend.trace.ops.unary_elementwise import exp

    exp_inp = exp(input - max(input, dim, keepdim=True))
    return exp_inp / sum(exp_inp, dim, keepdim=True)
