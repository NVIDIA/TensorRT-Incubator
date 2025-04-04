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

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Tuple

import nvtripy as tp
import pytest
import torch
from tests import helper


@dataclass
class ConvTransposeTestCase:
    num_spatial_dims: int
    stride: Sequence[int] = None
    padding: Sequence[Tuple[int, int]] = None
    dilation: Sequence[int] = None
    groups: int = 1
    bias: bool = False
    spatial_size: int = 2

    def __post_init__(self):
        self.padding = self.padding or [(0, 0)] * self.num_spatial_dims
        self.stride = self.stride or [1] * self.num_spatial_dims
        self.dilation = self.dilation or [1] * self.num_spatial_dims


TEST_CASES = [
    ConvTransposeTestCase(1),
    ConvTransposeTestCase(2),
    ConvTransposeTestCase(2, stride=(2, 2)),
    ConvTransposeTestCase(2, padding=[(1, 1), (1, 1)]),
    ConvTransposeTestCase(2, padding=[(1, 0), (0, 2)]),  # Asymmetric padding
    ConvTransposeTestCase(2, dilation=(2, 2), spatial_size=3),
    ConvTransposeTestCase(2, groups=2),
    ConvTransposeTestCase(2, bias=True),
    ConvTransposeTestCase(3),
]


@pytest.mark.parametrize("case", TEST_CASES)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
def test_conv_transpose(case, dtype, eager_or_compiled):
    IN_BATCH = 1
    IN_CHANNELS = 2
    IN_SPATIAL_DIMS = (case.spatial_size,) * case.num_spatial_dims
    OUT_CHANNELS = 2
    KERNEL_SIZE = (2,) * case.num_spatial_dims

    TorchConvTranspose = {1: torch.nn.ConvTranspose1d, 2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}[
        case.num_spatial_dims
    ]
    torch_dtype = helper.TORCH_DTYPES[dtype]

    inp_shape = (IN_BATCH, IN_CHANNELS) + IN_SPATIAL_DIMS

    with torch.no_grad():
        inp = torch.arange(math.prod(inp_shape), dtype=torch_dtype, device=torch.device("cuda")).reshape(inp_shape)

        torch_padding = []
        # Torch padding starts from the last dimension
        for tup in reversed(case.padding):
            torch_padding.extend(tup)

        torch_conv_transpose = TorchConvTranspose(
            IN_CHANNELS,
            OUT_CHANNELS,
            KERNEL_SIZE,
            case.stride,
            dilation=case.dilation,
            groups=case.groups,
            bias=case.bias,
            dtype=torch_dtype,
            device=torch.device("cuda"),
        )
        weights_shape = (IN_CHANNELS, OUT_CHANNELS // case.groups) + KERNEL_SIZE
        torch_conv_transpose.weight = torch.nn.Parameter(
            torch.arange(math.prod(weights_shape), dtype=torch_dtype, device=torch.device("cuda")).reshape(
                weights_shape
            ),
            requires_grad=False,
        )
        if case.bias:
            torch_conv_transpose.bias = torch.nn.Parameter(
                torch.arange(OUT_CHANNELS, dtype=torch_dtype, device=torch.device("cuda")), requires_grad=False
            )

        torch_out = torch_conv_transpose(inp)
        # torch.nn.ConvTranspose* do not support asymmetric padding, so we need to do that separately.
        # Padding in ConvTranspose is the same as cropping the output.
        slices = [slice(None), slice(None)]
        for low, high in case.padding:
            slices.append(slice(low, -high if high != 0 else None))
        torch_out = torch_out.__getitem__(slices).contiguous().clone()

    tripy_conv_transpose = tp.ConvTranspose(
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        case.stride,
        case.padding,
        case.dilation,
        case.groups,
        case.bias,
        dtype=dtype,
    )
    tripy_conv_transpose.weight = tp.Tensor(torch_conv_transpose.weight.to("cpu"))
    if case.bias:
        tripy_conv_transpose.bias = tp.Tensor(torch_conv_transpose.bias.to("cpu"))

    tripy_out = eager_or_compiled(tripy_conv_transpose, tp.Tensor(inp))

    torch_out = tp.Tensor(torch_out)
    assert tp.allclose(tripy_out, torch_out)
