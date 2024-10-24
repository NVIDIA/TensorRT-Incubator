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

from dataclasses import dataclass

from tripy import export, utils
from tripy.common import datatype
from tripy.frontend.module.module import Module
from tripy.frontend.module.parameter import DefaultParameter, Parameter

@export.public_api(document_under="operations/modules")
@dataclass
@utils.constant_fields(["num_features"])
class BatchNorm(Module):
    r"""
    Applies Batch Normalization over an N-dimensional input tensor using precomputed statistics.

    This implementation supports 1D, 2D, and 3D inputs (e.g., time-series, images, and volumetric data).
    Batch Normalization normalizes across the specified feature dimension (typically the second dimension in the input).
    The mean (:math:`\mu`) and variance (:math:`\sigma^2`) used are the precomputed running statistics.

    :math:`y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} * \gamma + \beta`

    where:
    - :math:`\mu` is the precomputed running mean.
    - :math:`\sigma^2` is the precomputed running variance.
    - :math:`\gamma` and :math:`\beta` are learnable parameter vectors of size `num_features`.

    This module is designed for evaluation purposes only, and it does not compute batch statistics.

    Attributes:
        num_features: The number of features or channels in the input (the size of the second dimension).
        eps: A small value added to the denominator for numerical stability.

    Example:

    .. code-block:: python
        :linenos:
        :caption: Example

        batch_norm = tp.BatchNorm(64)

        # For a 2D image tensor of shape (N, C, H, W)
        input = tp.iota((8, 64, 32, 32))
        output = batch_norm(input)

        # For a 1D signal tensor of shape (N, C, L)
        input_1d = tp.iota((16, 64, 128))
        output_1d = batch_norm(input_1d)

        # For a 3D volumetric tensor of shape (N, C, D, H, W)
        input_3d = tp.iota((4, 64, 16, 32, 32))
        output_3d = batch_norm(input_3d)
    """

    num_features: int
    r"""The number of feature channels in the input tensor (the size of the second dimension)."""

    weight: Parameter
    r"""The learnable scale parameter (γ) of shape `(num_features,)`."""

    bias: Parameter
    r"""The learnable shift parameter (β) of shape `(num_features,)`."""

    running_mean: Parameter
    r"""The precomputed running mean for the feature channels of shape `(num_features,)`."""

    running_var: Parameter
    r"""The precomputed running variance for the feature channels of shape `(num_features,)`."""

    eps: float = 1e-5
    r"""A small value added to the denominator to prevent division by zero during normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        # Initialize learnable parameters (scale and shift)
        self.weight = DefaultParameter((num_features,), dtype=datatype.float32)
        self.bias = DefaultParameter((num_features,), dtype=datatype.float32)

        # Initialize running statistics (precomputed, not updated)
        self.running_mean = DefaultParameter((num_features,), dtype=datatype.float32)
        self.running_var = DefaultParameter((num_features,), dtype=datatype.float32)

    def __call__(self, x: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            x: The input tensor with shape :math:`(N, C, ...)`, where C is the feature dimension.

        Returns:
            A tensor of the same shape as the input, with batch normalization applied using the precomputed running mean and variance.
        """
        from tripy.frontend.trace.ops.unary_elementwise import rsqrt
        from tripy.frontend.trace.ops.reshape import reshape

        x_shape = (1, -1, *([1] * (len(x.shape) - 2)))
        # Use precomputed running mean and variance for normalization
        mean = reshape(self.running_mean, x_shape)
        var = reshape(self.running_var, x_shape)

        # Normalize the input
        normalized = (x - mean) * rsqrt(var + self.eps)

        # Apply the learned scaling (weight) and shifting (bias)
        weight = reshape(self.weight, x_shape)
        bias =  reshape(self.bias, x_shape)
        
        return weight * normalized + bias
