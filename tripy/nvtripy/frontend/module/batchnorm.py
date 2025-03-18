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
from nvtripy.frontend.module.module import Module
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor


@export.public_api(document_under="operations/modules")
@dataclass
@utils.wrappers.constant_fields(["num_features"])
class BatchNorm(Module):
    r"""
    Applies batch normalization over an N-dimensional input tensor using precomputed statistics:

    :math:`y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta`

    where:
        - :math:`\mu` is the precomputed running mean.
        - :math:`\sigma^2` is the precomputed running variance.
        - :math:`\gamma` and :math:`\beta` are learnable parameter vectors (wieight and bias).

    This implementation supports 1D, 2D, and 3D inputs (e.g., time-series, images, and volumetric data).
    Batch Normalization normalizes across the specified feature dimension (typically the second dimension in the input).

    """

    num_features: int
    r"""The number of feature channels in the input tensor (the size of the second dimension)."""

    dtype: datatype.dtype
    r"""The data type used to perform the operation."""

    eps: float
    r""":math:`\epsilon` value added to the denominator to prevent division by zero during normalization."""

    weight: Tensor
    r"""The :math:`\gamma` parameter of shape :math:`[\text{num_features}]`."""

    bias: Tensor
    r"""The :math:`\beta` parameter of shape :math:`[\text{num_features}]`."""

    running_mean: Tensor
    r"""The running mean for the feature channels of shape :math:`[\text{num_features}]`."""

    running_var: Tensor
    r"""The running variance for the feature channels of shape :math:`[\text{num_features}]`."""

    def __init__(self, num_features: int, dtype: datatype.dtype = datatype.float32, eps: float = 1e-5) -> None:
        r"""
        Args:
            num_features: The number of feature channels in the input tensor (the size of the second dimension).
            dtype: The data type to use for the weight, bias, running_mean and running_var parameters.
            eps: :math:`\epsilon` value added to the denominator to prevent division by zero during normalization.

        .. code-block:: python
            :linenos:

            batch_norm = tp.BatchNorm(2)

            batch_norm.weight = tp.iota(batch_norm.weight.shape)
            batch_norm.bias = tp.iota(batch_norm.bias.shape)
            batch_norm.running_mean = tp.iota(batch_norm.running_mean.shape)
            batch_norm.running_var = tp.iota(batch_norm.running_var.shape)

            input = tp.iota((1, 2, 1, 1))
            output = batch_norm(input)
        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        # Initialize learnable parameters (scale and shift)
        self.weight = DefaultParameter((num_features,), dtype=dtype)
        self.bias = DefaultParameter((num_features,), dtype=dtype)

        # Initialize running statistics (precomputed, not updated)
        self.running_mean = DefaultParameter((num_features,), dtype=dtype)
        self.running_var = DefaultParameter((num_features,), dtype=dtype)

    def forward(self, x: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            x: The input tensor with shape :math:`(N, C, ...)`, where C is the feature dimension.

        Returns:
            A tensor of the same shape as the input.
        """
        from nvtripy.frontend.ops.unary.rsqrt import rsqrt
        from nvtripy.frontend.ops.reshape import reshape

        x_shape = (1, self.num_features, *([1] * (x.rank - 2)))

        # Use precomputed running mean and variance for normalization
        mean = reshape(self.running_mean, x_shape)
        var = reshape(self.running_var, x_shape)

        # Normalize the input
        x = (x - mean) * rsqrt(var + self.eps)

        # Apply the learned scaling (weight) and shifting (bias)
        weight = reshape(self.weight, x_shape)
        bias = reshape(self.bias, x_shape)

        return weight * x + bias
