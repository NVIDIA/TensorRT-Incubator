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

from dataclasses import dataclass

from tripy import export, utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import Parameter, DefaultParameter


@export.public_api(document_under="modules")
@dataclass
@utils.constant_fields(["dtype"])
class LayerNorm(Module):
    r"""
    Applies layer normalization over the input tensor:

    :math:`\text{LayerNorm}(x) = \Large \frac{x - \bar{x}}{ \sqrt{\sigma^2 + \epsilon}} \normalsize * \gamma + \beta`

    where :math:`\bar{x}` is the mean and :math:`\sigma^2` is the variance.
    """

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Parameter
    r"""The :math:`\gamma` parameter of shape :math:`[\text{normalized_shape}]`."""

    bias: Parameter
    r"""The :math:`\beta` parameter of shape :math:`[\text{normalized_shape}]`."""

    eps: float
    """A value added to the denominator to prevent division by zero."""

    def __init__(self, normalized_shape: int, dtype: datatype.dtype = datatype.float32) -> None:
        """
        Args:
            normalized_shape: The size of the feature dimension of the input over which normalization is performed.
            dtype: The data type to use for the weight and bias parameters.

        .. code-block:: python
            :linenos:
            :caption: Example

            layer_norm = tp.LayerNorm(3)

            input = tp.iota((2, 3))
            output = layer_norm(input)

            np_out = cp.from_dlpack(output).get() # doc: omit
            assert np_out.shape == (2, 3)

            torch_tensor = torch.from_dlpack(input) # doc: omit
            torch_ln = torch.nn.LayerNorm(3) # doc: omit
            torch_ln.weight.data = torch.from_dlpack(layer_norm.weight) # doc: omit
            torch_ln.bias.data = torch.from_dlpack(layer_norm.bias) # doc: omit
            assert np.allclose(np_out, cp.from_dlpack(torch_ln(torch_tensor).detach()).get())
        """
        super().__init__()

        self.dtype = dtype

        # Replace with random weights when #74 is completed.
        self.weight = DefaultParameter((normalized_shape,), dtype=dtype)

        self.bias = DefaultParameter((normalized_shape,), dtype=dtype)

        self.eps = 1e-5

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor.

        Returns:
            A tensor of the same shape as the input.
        """
        from tripy.frontend.trace.ops.reduce import mean, var
        from tripy.frontend.trace.ops.unary_elementwise import rsqrt

        mean_val = mean(x, dim=-1, keepdim=True)
        var_val = var(x, dim=-1, keepdim=True) + self.eps
        x = (x - mean_val) * rsqrt(var_val)
        return self.weight * x + self.bias
