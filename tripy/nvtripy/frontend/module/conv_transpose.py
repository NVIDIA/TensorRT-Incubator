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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from nvtripy import export
from nvtripy.common import datatype
from nvtripy.frontend.module.conv import ConvBase
from nvtripy.frontend.module.parameter import DefaultParameter
from nvtripy.frontend.tensor import Tensor


@export.public_api(document_under="operations/modules", autodoc_options=[":no-show-inheritance:"])
@dataclass
class ConvTranspose(ConvBase):
    r"""
    Applies a transposed convolution operation on the input tensor.

    Transposed convolution, also known as fractionally-strided convolution or deconvolution, performs a "reverse" of a
    standard convolution. It upsamples the input to a larger spatial resolution, such that if you were to apply a
    standard convolution and then a transpose convolution with the same parameters, you would get back the original spatial dimensions.

    The transposed convolution operation can be thought of as a regular convolution operation applied to a dilated
    (i.e. zeros are inserted between the input values) version of the input tensor. The stride parameter controls the
    dilation factor, and the padding effectively indicates how much to crop from the output.

    Note that transposed convolution is not a strict inverse of standard convolution.
    """

    dtype: datatype.dtype
    r"""The data type to use for the convolution weights."""

    weight: Tensor
    r"""
    The kernel of shape :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}}, *\text{kernel_dims})`.
    """

    padding: Sequence[Sequence[int]]
    r"""
    A sequence of pairs of integers of length :math:`M` indicating the implicit zero padding
    applied along each spatial dimension before and after the dimension respectively,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`. In particular,
    :math:`\text{dilation} \times (\text{kernel_dims}_i - 1) - \text{padding}_i` will be added to or cropped from the input.
    This is set so that when this module is initialized with the same parameters as :class:`nvtripy.Conv`,
    they are inverses with respect to the input/output shapes.
    """

    stride: Sequence[int]
    r"""
    A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
    where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
    For transposed convolution, this effectively controls the dilation of the input; for each dimension with
    value :math:`x`, :math:`x-1` zeros are inserted between input values.
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
        padding: Optional[Sequence[Sequence[int]]] = None,
        stride: Optional[Sequence[int]] = None,
        groups: Optional[int] = None,
        dilation: Optional[Sequence[int]] = None,
        bias: bool = True,
        dtype: datatype.dtype = datatype.float32,
    ) -> None:
        r"""
        Args:
            in_channels: The number of channels in the input tensor.
            out_channels: The number of channels produced by the convolution.
            kernel_dims: The spatial shape of the kernel.
            padding: A sequence of pairs of integers of length :math:`M` indicating the implicit zero padding
                applied along each spatial dimension before and after the dimension respectively,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`. In particular,
                :math:`\text{dilation} \times (\text{kernel_dims}_i - 1) - \text{padding}_i` will be added to or cropped from the input.
                This is set so that when this module is initialized with the same parameters as :class:`nvtripy.Conv`,
                they are inverses with respect to the input/output shapes. Defaults to all 0.
            stride: A sequence of length :math:`M` indicating the stride of convolution across each spatial dimension,
                where :math:`M` is the number of spatial dimensions, i.e. :math:`M = \text{rank(input)} - 2`.
                For transposed convolution, this effectively controls the dilation of the input; for each dimension with
                value :math:`x`, :math:`x-1` zeros are inserted between input values. Defaults to all 1.
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
            :caption: Example

            input = tp.reshape(tp.arange(4, dtype=tp.float32), (1, 1, 2, 2))
            upsample = tp.ConvTranspose(1, 1, (3, 3), stride=(2, 2), bias=False, dtype=tp.float32)
            output = upsample(input)

            conv_layer_torch = torch.nn.ConvTranspose2d(1, 1, 3, stride=2, bias=False, dtype=torch.float32, device=torch.device("cuda")) # doc: omit
            conv_layer_torch.weight.data = torch.from_dlpack(upsample.weight) # doc: omit
            expected = conv_layer_torch(torch.from_dlpack(input)) # doc: omit

            output_torch = torch.from_dlpack(output) # doc: omit
            assert torch.allclose(output_torch, expected)
            assert output_torch.shape == expected.shape

        .. code-block:: python
            :linenos:
            :caption: "Inversing" Convolution

            # This process restores the input spatial dimensions, but not its values
            input = tp.reshape(tp.arange(16, dtype=tp.float32), (1, 1, 4, 4))
            downsample = tp.Conv(1, 1, (2, 2), stride=(2, 2), padding=((1, 1), (1, 1)), bias=False, dtype=tp.float32 )
            upsample = tp.ConvTranspose(1, 1, (2, 2), stride=(2, 2), padding=((1, 1), (1, 1)), bias=False, dtype=tp.float32)
            output_down = downsample(input)
            output_up = upsample(output_down)

            downsample_torch = torch.nn.Conv2d(1, 1, 2, stride=2, padding=1, bias=False, dtype=torch.float32) # doc: omit
            downsample_torch.weight.data = torch.from_dlpack(downsample.weight) # doc: omit
            expected_down = downsample_torch(torch.from_dlpack(input)) # doc: omit
            upsample_torch = torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=1, bias=False, dtype=torch.float32) # doc: omit
            upsample_torch.weight.data = torch.from_dlpack(upsample.weight) # doc: omit
            expected_up = upsample_torch(expected_down) # doc: omit

            output_down_torch = torch.from_dlpack(output_down) # doc: omit
            output_up_torch = torch.from_dlpack(output_up) # doc: omit
            assert torch.allclose(output_down_torch, expected_down)
            assert output_down_torch.shape == expected_down.shape
            assert torch.allclose(output_up_torch, expected_up)
            assert output_up_torch.shape == expected_up.shape
        """
        # TODO: add `output_padding` field and test if it works with SHLO/MLIR-TRT

        super().__init__(in_channels, out_channels, kernel_dims, padding, stride, groups, dilation, bias, dtype)

        kernel_shape = (in_channels, out_channels // self.groups, *kernel_dims)
        self.weight = DefaultParameter(kernel_shape, dtype=dtype)
        self._kernel_hlo_intermediate_shape = (
            self.groups,
            in_channels // self.groups,
            out_channels // self.groups,
            *kernel_dims,
        )
        self._kernel_hlo_final_shape = (out_channels, in_channels // self.groups, *kernel_dims)

        rank = len(kernel_shape)
        self._dummy_stride = (1,) * (rank - 2)

        # Adjust padding to create inverse input/output shape with same params to Conv.
        # User-provided padding effectively removes from the input, and is restored in MLIR-TRT.
        transpose_padding = []
        for pad, dilation, kernel_size in zip(self.padding, self.dilation, self._kernel_hlo_final_shape[2:]):
            transpose_padding.append(
                (
                    dilation * (kernel_size - 1) - pad[0],
                    dilation * (kernel_size - 1) - pad[1],
                )
            )
        self._transpose_padding = tuple(transpose_padding)

    def __call__(self, input: "nvtripy.Tensor") -> "nvtripy.Tensor":
        r"""
        Args:
            input: The input tensor.

        Returns:
            A tensor of the same data type as the input with a shape
            :math:`(N, \text{out_channels}, D_{0_{\text{out}}},\ldots,D_{n_{\text{out}}})`
            where :math:`D_{k_{\text{out}}} = (D_{k_{\text{in}}} - 1) \times \text{stride}_k - \text{padding}_{k_0} - \text{padding}_{k_1} + \text{dilation}_k \times (\text{kernel_dims}_k - 1) + 1`
        """
        from nvtripy.frontend.ops.transpose import transpose
        from nvtripy.frontend.trace.ops.convolution import convolution
        from nvtripy.frontend.trace.ops.flip import flip
        from nvtripy.frontend.trace.ops.reshape import reshape

        # SHLO expects kernel shape in (out_channels, in_channels / feature_groups, ...) format
        # whereas typically transpose conv uses (in_channels, out_channels / feature_groups, ...) e.g. in Torch.
        # We enforce SHLO shape expectation here, and MLIR-TRT reshapes to [i, o/feature_group, ...] format for TRT.
        # The spatial dimensions must also be flipped to match expectations for conv/deconv in MLIR-TRT.
        rank = self.weight.rank
        weight = flip(self.weight, list(range(2, rank)))
        if self.groups == 1:
            weight = transpose(weight, 0, 1)
        else:
            weight = reshape(weight, self._kernel_hlo_intermediate_shape)
            weight = transpose(weight, 1, 2)
            weight = reshape(weight, self._kernel_hlo_final_shape)

        x = convolution(
            input,
            weight,
            self._transpose_padding,
            self._dummy_stride,
            self.groups,
            self.stride,  # effectively lhs_dilation for StableHLO
            self.dilation,
        )
        if self.bias is not None:
            bias_shape_to_broadcast = (1, weight.shape[0]) + (1,) * (rank - 2)
            x += reshape(self.bias, bias_shape_to_broadcast)
        return x
