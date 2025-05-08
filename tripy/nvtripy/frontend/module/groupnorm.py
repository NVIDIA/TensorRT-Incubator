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

from dataclasses import dataclass

from nvtripy import export, utils
from nvtripy.common import datatype
from nvtripy.common.exception import raise_error
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor

from nvtripy.frontend.ops import utils as op_utils
from nvtripy.utils import wrappers
from nvtripy.trace.ops.groupnorm import GroupNorm as GroupNormOp


@wrappers.interface(
    dtype_constraints={"input": "T1", "weight": "T1", "bias": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def group_norm(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: "nvtripy.Tensor",
    num_groups: int,
    num_channels: int,
    eps: float = 1e-5,
) -> "nvtripy.Tensor":
    from nvtripy.frontend.ops.reshape import reshape
    from nvtripy.frontend.ops.reduce.mean import mean

    # Reshape weight/bias from [num_channels] to [num_groups] by averaging over channels_per_group
    channels_per_group = num_channels // num_groups
    weight_reshaped = reshape(weight, (num_groups, channels_per_group))
    bias_reshaped = reshape(bias, (num_groups, channels_per_group))
    weight_grouped = mean(weight_reshaped, dim=1)
    bias_grouped = mean(bias_reshaped, dim=1)

    if input.rank < 3:
        raise_error(
            "Input is expected to have shape [N, C, D1, ...] where N is the batch size, C is the number of channels, and D1, D2, ... are the feature dimensions.",
            details=[f"Got {input.shape} which has rank {input.rank} < 3."],
        )

    # TensorRT expects scale and bias to have shape [1, G, 1, 1, ..., 1]
    broadcast_shape = (1, num_groups) + (1,) * (input.rank - 2)
    weight = reshape(weight_grouped, broadcast_shape)
    bias = reshape(bias_grouped, broadcast_shape)

    # Set the shape information in the trace_tensors in case of dynamic shapes
    weight.trace_tensor.shape = broadcast_shape
    bias.trace_tensor.shape = broadcast_shape
    input.trace_tensor.shape = input.trace_tensor.shape[:1] + (num_channels,) + input.trace_tensor.shape[2:]

    return op_utils.create_op(
        GroupNormOp,
        [input, weight, bias],
        num_groups=num_groups,
        eps=eps,
    )


@export.public_api(document_under="operations/modules")
@dataclass
@utils.wrappers.constant_fields(["num_groups", "num_channels", "dtype"])
class GroupNorm(Module):
    r"""
    Applies group normalization over the input tensor:

    :math:`\text{GroupNorm}(x) = \Large \frac{x - \bar{x}}{ \sqrt{\sigma^2 + \epsilon}} \normalsize * \gamma + \beta`

    where :math:`\bar{x}` is the mean and :math:`\sigma^2` is the variance.

    The input should have shape :math:`[N, C, D1, ...]` where :math:`N` is the batch size, :math:`C` is the number of channels, and :math:`D1, ...` are the feature dimensions.
    """

    num_groups: int
    r"""The number of groups to split the channels into."""

    num_channels: int
    """The number of channels expected in the input."""

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    weight: Tensor
    r"""The :math:`\gamma` parameter of shape :math:`[\text{num_channels}]`."""

    bias: Tensor
    r"""The :math:`\beta` parameter of shape :math:`[\text{num_channels}]`."""

    eps: float
    """A value added to the denominator to prevent division by zero. Defaults to 1e-5."""

    def __init__(
        self, num_groups: int, num_channels: int, dtype: datatype.dtype = datatype.float32, eps: float = 1e-5
    ) -> None:
        r"""
        Args:
            num_groups: The number of groups to split the channels into.
            num_channels: The number of channels expected in the input.
            dtype: The data type to use for the weight and bias parameters.
            eps: :math:`\epsilon` value to prevent division by zero.

        .. code-block:: python
            :linenos:

            group_norm = tp.GroupNorm(2, 2)

            group_norm.weight = tp.iota(group_norm.weight.shape)
            group_norm.bias = tp.iota(group_norm.bias.shape)

            input = tp.iota((1, 2, 2, 2), dim=1)
            output = group_norm(input)

            np_out = cp.from_dlpack(output).get() # doc: omit
            assert np_out.shape == (1, 2, 2, 2)

            torch_tensor = torch.from_dlpack(input) # doc: omit
            torch_gn = torch.nn.GroupNorm(2, 2).to(torch.device("cuda")) # doc: omit
            torch_gn.weight.data = torch.from_dlpack(group_norm.weight) # doc: omit
            torch_gn.bias.data = torch.from_dlpack(group_norm.bias) # doc: omit
            torch_out = cp.from_dlpack(torch_gn(torch_tensor).detach()).get() # doc: omit
            assert np_out.shape == torch_out.shape # doc: omit
            assert np.allclose(np_out, torch_out) # doc: omit
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
        self.eps = eps

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: The input tensor.

        Returns:
            A tensor of the same shape as the input.
        """
        return group_norm(x, self.weight, self.bias, self.num_groups, self.num_channels, self.eps)
