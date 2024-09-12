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
from collections.abc import Sequence

from tripy import export, utils
from tripy.frontend.module.module import Module

from tripy.common.exception import raise_error


@export.public_api(document_under="operations/modules")
@dataclass
@utils.constant_fields(["dtype", "normalized_shape"])
class MaxPool(Module):
    r"""
    Applies a max pooling over the input tensor.

    With an input of shape :math:`(N, C_{\text{in}}, D_0,\ldots,D_n)` and
    output of shape :math:`(N, C_{\text{out}}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
    the output values are given by:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{Bias}_{C_{\text{out}}} +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the cross-correlation operator applied over the spatial
    dimensions of the input and kernel,
    :math:`N` is the batch dimension, :math:`C` is the channel dimension, and
    :math:`D_0,\ldots,D_n` are the spatial dimensions.
    """

    kernel_dims: Sequence[int]
    r"""
    A sequence of integers representing the window sizes of the pooling operation.
    """

    stride: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    padding: Sequence[Sequence[int]]
    r"""
    A sequence of pairs of integers of length :math:`M` indicating the zero padding
    to apply to the input along each spatial dimension before and after the dimension respectively,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    def __init__(
        self,
        kernel_dims: Sequence[int],
        stride: Sequence[int] = None,
        padding: Sequence[Sequence[int]] = None,
    ) -> None:
        r"""
        Args:
            kernel_dims: The spatial shape of the pooling window.
            stride: A sequence of length :math:`M` indicating the stride of pooling across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 1.
            padding: A sequence of pairs of integers of length :math:`M` indicating the zero padding
                to apply to the input along each spatial dimension before and after the dimension respectively,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 0.

        .. code-block:: python
            :linenos:
            :caption: Example

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            pool = tp.MaxPool(kernel_dims=(2, 2))
            output = pool(input)

            pool_torch = torch.nn.MaxPool2d((2, 2)) # doc: omit
            expected = pool_torch(torch.from_dlpack(input)) # doc: omit

            assert torch.allclose(torch.from_dlpack(output), expected)
        """
        super().__init__()

        spatial_dims = len(kernel_dims)
        self.kernel_dims = kernel_dims
        if stride is not None:
            if len(stride) != spatial_dims:
                raise_error(
                    "Stride must have the same length as kernel_dims.",
                    [f"Got stride={stride}, ", f"kernel_dims={kernel_dims}"],
                )
            self.stride = stride
        else:
            self.stride = [1] * spatial_dims
        if padding is not None:
            if len(padding) != spatial_dims:
                raise_error(
                    "Padding must have the same length as kernel_dims.",
                    [f"Got padding={padding}, ", f"kernel_dims={kernel_dims}"],
                )
            if any(len(pad_size) != 2 for pad_size in padding):
                raise_error(
                    f"Padding must be provided as a sequence of pairs of integers.",
                    details=[
                        f"Supplied padding attribute: {self.padding} contains sequences that are not of length 2."
                    ],
                )
            self.padding = padding
        else:
            self.padding = [(0, 0)] * spatial_dims

    def __call__(self, input: "tripy.Tensor") -> "tripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            Result tensor after pooling.
        """
        from tripy.frontend.trace.ops.pooling import maxpool

        x = maxpool(input, self.kernel_dims, self.stride, self.padding)
        return x
