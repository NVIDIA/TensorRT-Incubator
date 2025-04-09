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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Tuple

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.frontend.module.conv.base import ConvBase
from nvtripy.frontend.module.conv.utils import conv_deconv_helper
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor
from nvtripy.trace.ops.convolution import Convolution
from nvtripy.utils import wrappers


# This function is added so that we can do dtype checking.
@wrappers.interface(
    dtype_constraints={"input": "T1", "weight": "T1", "bias": "T1", wrappers.RETURN_VALUE: "T1"},
    dtype_variables={"T1": ["float32", "float16", "bfloat16"]},
)
def convolution(
    input: "nvtripy.Tensor",
    weight: "nvtripy.Tensor",
    bias: Optional["nvtripy.Tensor"],
    stride: Sequence[int],
    padding: Sequence[Sequence[int]],
    groups: int,
    dilation: Sequence[int],
):
    out = conv_deconv_helper(Convolution, input, weight, bias, stride, padding, groups, dilation)
    # Encode as much information about the output shape as we can:
    out_shape = list(out.trace_tensor.shape)
    out_shape[1] = weight.trace_tensor.shape[0]
    out.trace_tensor.shape = tuple(out_shape)
    return out


@export.public_api(document_under="operations/modules", autodoc_options=[":no-show-inheritance:"])
@dataclass
class Conv(ConvBase):
    r"""
    Applies a convolution on the input tensor.

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

    dtype: datatype.dtype
    r"""The data type to use for the convolution weights."""

    weight: Tensor
    r"""The kernel of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, *\text{kernel_dims})`."""

    padding: Sequence[Tuple[int, int]]
    r"""
    A sequence of pairs of integers of length :math:`M` indicating the zero padding
    to apply to the input along each spatial dimension before and after the dimension respectively,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    stride: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    """

    groups: int
    r"""
    The number of groups in a grouped convolution where the input and output channels are divided into ``groups`` groups.
    Each output group is connected only to its corresponding input group through the convolution kernel weights,
    and the outputs for each group are concatenated to produce the final result. This is in contrast to a standard convolution
    which has full connectivity between all input and output channels. Grouped convolutions reduce computational cost by
    a factor of ``groups`` and can benefit model parallelism and memory usage.
    Note that `in_channels` and `out_channels` must both be divisible by ``groups``.
    """

    dilation: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the number of zeros to insert between kernel weights across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    This is known as the a trous algorithm and further downsamples the output by increasing the receptive field of the kernel.
    For each dimension with value :math:`x`, :math:`x-1` zeros are inserted between kernel weights.
    """

    bias: Optional[Tensor]
    r"""
    The bias term to add to the output. The bias has a shape of :math:`(\text{out_channels},)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_dims: Sequence[int],
        stride: Optional[Sequence[int]] = None,
        padding: Optional[Sequence[Tuple[int, int]]] = None,
        dilation: Optional[Sequence[int]] = None,
        groups: Optional[int] = None,
        bias: bool = True,
        dtype: datatype.dtype = datatype.float32,
    ) -> None:
        r"""
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels produced by the convolution.
            kernel_dims: The spatial shape of the kernel.
            padding: A sequence of pairs of integers of length :math:`M` indicating the zero padding
                to apply to the input along each spatial dimension before and after the dimension respectively,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 0.
            stride: A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                Defaults to all 1.
            groups: The number of groups in a grouped convolution where the input and output channels are divided into ``groups`` groups.
                Each output group is connected only to its corresponding input group through the convolution kernel weights,
                and the outputs for each group are concatenated to produce the final result. This is in contrast to a standard convolution
                which has full connectivity between all input and output channels. Grouped convolutions reduce computational cost by
                a factor of ``groups`` and can benefit model parallelism and memory usage.
                Note that `in_channels` and `out_channels` must both be divisible by ``groups``. Defaults to 1 (standard convolution).
            dilation: A sequence of length :math:`M` indicating the number of zeros to insert between kernel weights across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                This is known as the a trous algorithm and further downsamples the output by increasing the receptive field of the kernel.
                For each dimension with value :math:`x`, :math:`x-1` zeros are inserted between kernel weights.
            bias: Whether to add a bias term to the output or not. The bias has a shape of :math:`(\text{out_channels},)`.
            dtype: The data type to use for the convolution weights.

        .. code-block:: python
            :linenos:

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(in_channels=1, out_channels=1, kernel_dims=(2, 2), dtype=tp.float32)

            conv.weight = tp.iota(conv.weight.shape)
            conv.bias = tp.iota(conv.bias.shape)

            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2) # doc: omit
            conv_layer_torch.weight.data = torch.from_dlpack(conv.weight) # doc: omit
            conv_layer_torch.bias.data = torch.from_dlpack(conv.bias).reshape([-1]) # doc: omit
            expected = conv_layer_torch(torch.from_dlpack(input)) # doc: omit

            assert torch.allclose(torch.from_dlpack(output), expected)

        .. code-block:: python
            :linenos:
            :caption: Using Padding and Stride

            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            conv = tp.Conv(1, 1, (3, 3), padding=((1, 1), (1, 1)), stride=(3, 1), bias=False, dtype=tp.float32)

            conv.weight = tp.iota(conv.weight.shape)

            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, padding=1, stride=(3, 1), bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_dlpack(conv.weight) # doc: omit
            expected = conv_layer_torch(torch.from_dlpack(input)) # doc: omit

            assert torch.allclose(torch.from_dlpack(output), expected)

        .. code-block:: python
            :linenos:
            :caption: Depthwise Convolution

            input = tp.reshape(tp.arange(18, dtype=tp.float32), (1, 2, 3, 3))
            conv = tp.Conv(2, 2, (3, 3), groups=2, bias=False, dtype=tp.float32)

            conv.weight = tp.iota(conv.weight.shape)

            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(2, 2, 3, groups=2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_dlpack(conv.weight) # doc: omit
            expected = conv_layer_torch(torch.from_dlpack(input)) # doc: omit

            assert torch.allclose(torch.from_dlpack(output), expected)

        .. code-block:: python
            :linenos:
            :caption: Dilated Convolution (a trous algorithm)

            input = tp.reshape(tp.arange(9, dtype=tp.float32), (1, 1, 3, 3))
            conv = tp.Conv(1, 1, (2, 2), dilation=(2, 2), bias=False, dtype=tp.float32)

            conv.weight = tp.iota(conv.weight.shape)

            output = conv(input)

            conv_layer_torch = torch.nn.Conv2d(1, 1, 2, dilation=2, bias=False) # doc: omit
            conv_layer_torch.weight.data = torch.from_dlpack(conv.weight) # doc: omit
            expected = conv_layer_torch(torch.from_dlpack(input)) # doc: omit

            assert torch.allclose(torch.from_dlpack(output), expected)
        """

        super().__init__(in_channels, out_channels, kernel_dims, padding, stride, groups, dilation, bias, dtype)

        kernel_shape = (out_channels, in_channels // self.groups, *kernel_dims)
        self.weight = DefaultParameter(kernel_shape, dtype=dtype)

    def forward(self, input: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor of the same data type as the input with a shape
            :math:`(N, \text{out_channels}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
            where :math:`D_{k_{\text{out}}} = \large \left\lfloor \frac{D_{k_{\text{in}}} + \text{padding}_{k_0} + \text{padding}_{k_1} - \text{dilation}_k \times (\text{kernel_dims}_k - 1) - 1}{\text{stride}_k} \right\rfloor + \normalsize 1`
        """
        return convolution(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.groups,
            self.dilation,
        )
