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

from tripy.common.exception import raise_error


@export.public_api(document_under="modules")
@dataclass
@utils.constant_fields(["num_groups", "num_channels", "dtype"])
class GroupNorm(Module):
    r"""
    Applies group normalization over the input tensor:

    :math:`\text{GroupNorm}(x) = \Large \frac{x - \bar{x}}{ \sqrt{\sigma^2 + \epsilon}} \normalsize * \gamma + \beta`

    where :math:`\bar{x}` is the mean and :math:`\sigma^2` is the variance.
    """

    num_groups: int
    r"""The number of groups to split the channels into."""

    num_channels: int
    """The number of channels expected in the input."""

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Parameter
    r"""The :math:`\gamma` parameter of shape :math:`[\text{num_channels}]`."""

    bias: Parameter
    r"""The :math:`\beta` parameter of shape :math:`[\text{num_channels}]`."""

    eps: float
    """A value added to the denominator to prevent division by zero. Defaults to 1e-5."""

    def __init__(self, num_groups: int, num_channels: int, dtype: datatype.dtype = datatype.float32) -> None:
        """
        Args:
            num_groups: The number of groups to split the channels into.
            num_channels: The number of channels expected in the input.
            dtype: The data type to use for the weight and bias parameters.

        .. code-block:: python
            :linenos:
            :caption: Example

            group_norm = tp.GroupNorm(2, 4)
            group_norm.weight = tp.ones_like(group_norm.weight)
            group_norm.bias = tp.zeros_like(group_norm.bias)

            input = tp.iota((1, 4, 2, 2))
            output = group_norm(input)

            np_out = cp.from_dlpack(output).get() # doc: omit
            assert np_out.shape == (1, 4, 2, 2)

            torch_tensor = torch.from_dlpack(input) # doc: omit
            torch_gn = torch.nn.GroupNorm(2, 4).to(torch.device("cuda")) # doc: omit
            assert np.array_equal(np_out, cp.from_dlpack(torch_gn(torch_tensor).detach()).get())
        """
        super().__init__()

        if num_channels % num_groups:
            raise_error(
                "Number of groups must divide number of channels evenly.",
                details=[f"Got {num_groups} groups but {num_channels} channels."],
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.dtype = dtype

        # Replace with random weights when #74 is completed.
        self.weight = DefaultParameter((num_channels,), dtype=dtype)
        self.bias = DefaultParameter((num_channels,), dtype=dtype)
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
        from tripy.frontend.trace.ops.reshape import reshape
        from tripy.frontend.trace.ops.unsqueeze import unsqueeze

        # TODO #95 & #174 - Get integers out of TP Array
        batch_size = int(x.shape[0].data().data())
        spatial_shape = [int(s) for s in x.shape[2:].data().data()]

        x = reshape(x, (batch_size, self.num_groups, -1))
        mean_val = mean(x, dim=-1, keepdim=True)
        var_val = var(x, dim=-1, keepdim=True) + self.eps
        x = (x - mean_val) * rsqrt(var_val)
        x = reshape(x, (batch_size, self.num_channels, *spatial_shape))

        shape_to_broadcast = (1,) + (self.num_channels,) + (1,) * (x.rank - 2)

        return reshape(self.weight, shape_to_broadcast) * x + reshape(self.bias, shape_to_broadcast)
