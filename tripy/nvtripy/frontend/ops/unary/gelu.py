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
from nvtripy.common import datatype as dt
from nvtripy.frontend import wrappers
from nvtripy.trace.ops.unary import GeluErf
from nvtripy.frontend.ops import utils as op_utils
from nvtripy.frontend.constraints import GetInput, GetReturn, OneOf


@export.public_api(document_under="operations/functions")
@wrappers.interface(
    input_requirements=OneOf(GetInput("input").dtype, [dt.float32, dt.float16, dt.bfloat16, dt.int8]),
    output_guarantees=GetReturn(0).dtype == GetInput("input").dtype,
)
def gelu(input: "nvtripy.Tensor") -> "nvtripy.Tensor":
    r"""
    Applies a Gaussian Error Linear Unit (GELU) function
    to each element of the input tensor:

    :math:`\text{gelu}(x) = x * \phi(x)`

    where :math:`\phi(x)` is the
    `Cumulative Distribution Function (CDF) of a Gaussian distribution <https://en.wikipedia.org/wiki/Normal_distribution#Cumulative_distribution_function>`_.

    Args:
        input: The input tensor.

    Returns:
        A tensor of the same shape as the input.

    .. code-block:: python
        :linenos:

        input = tp.Tensor([1., 2., 3., 4.])
        output = tp.gelu(input)

        t = torch.tensor([1, 2, 3, 4], dtype=torch.float32) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch.nn.functional.gelu(t)))
    """
    return op_utils.create_op(GeluErf, [input])
