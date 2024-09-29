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

from tripy import export, constraints


@export.public_api(document_under="operations/functions")
@constraints.dtype_info(
    dtype_variables={"T1": ["float32", "float16", "bfloat16", "float8"]},
    dtype_constraints={"abs": "T1", "angles": "T1", constraints.RETURN_VALUE: "T1"},
)
def cartesian_via_polar(abs: "tripy.Tensor", angles: "tripy.Tensor") -> "tripy.Tensor":
    r"""
    Constructs the real-valued cartesian coordinates from magnitude and angle representing polar coordinates. For input
    ``abs`` and ``angle`` of shape :math:`(m_1, m_2, \ldots, m_i),` this function returns a new real tensor of shape
    :math:`(m_1, m_2, \ldots, m_i, 2)` where the last dimension of size 2 represents the real and imaginary components.

    .. math::
        \text{out}[\ldots, 0] = \text{abs} * \cos(\text{angle}) \\
        \text{out}[\ldots, 1] = \text{abs} * \sin(\text{angle})

    Args:
        abs: Tensor of absolute values.
        angles: Tensor of angles in radians.

    Returns:
        A real-valued tensor representing the Cartesian coordinates from the input polar coordinates.

    .. code-block:: python
        :linenos:
        :caption: Example

        import math

        abs = tp.Tensor([1.0, 2.0, 3.0])
        angles = tp.Tensor([0.0, math.pi/4, math.pi/2])
        output = tp.cartesian_via_polar(abs, angles)

        torch_abs = torch.Tensor([1., 2., 3.]) # doc: omit
        torch_angles = torch.Tensor([0.0, math.pi/4, math.pi/2]) # doc: omit
        torch_out = torch.view_as_real(torch.polar(torch_abs, torch_angles)) # doc: omit
        assert tp.allclose(output, tp.Tensor(torch_out))
        assert output.shape == torch_out.shape
    """
    from tripy.frontend.trace.ops.unary_elementwise import sin, cos
    from tripy.frontend.ops.stack import stack

    # Convert polar to complex
    real = abs * cos(angles)
    imag = abs * sin(angles)

    # Stack real and imaginary parts
    return stack([real, imag], dim=-1)
