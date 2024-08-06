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

from collections.abc import Sequence
from dataclasses import dataclass

import pytest
import torch

import tripy as tp
from tests import helper

DTYPES = [
    (torch.float16, tp.float16),
    (torch.float32, tp.float32),
]


@dataclass
class ConvTestCase:
    tp_pad: Sequence[Sequence[int]] = None
    torch_pad: Sequence[int] = None
    stride: Sequence[int] = None
    groups: int = 1
    dilation: Sequence[int] = None
    bias: bool = False


test_cases_1d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((2, 2),), torch_pad=(2,)),
    ConvTestCase(stride=(2,)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(2,)),
    ConvTestCase(bias=True),
]

test_cases_2d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((2, 2), (1, 1)), torch_pad=(2, 1)),
    ConvTestCase(stride=(2, 1)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(1, 2)),
    ConvTestCase(bias=True),
]

test_cases_3d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((0, 0), (2, 2), (0, 0)), torch_pad=(0, 2, 0)),
    ConvTestCase(stride=(2, 1, 2)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(2, 2, 2)),
    ConvTestCase(bias=True),
]


# TODO (#147): Update tests to use Torch FP16 convolution
@pytest.mark.parametrize("torch_dtype,tp_dtype", DTYPES)
class TestConvolution:
    @pytest.mark.parametrize("test_case", test_cases_1d)
    def test_convolution_1d(self, torch_dtype, tp_dtype, test_case, compile_fixture):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1,)
        if not test_case.dilation:
            test_case.dilation = (1,)

        input_torch = torch.arange(40, dtype=torch.float32, device=torch.device("cuda")).reshape(*(2, 4, 5))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv1d(
            4,
            8,
            3,
            padding=test_case.torch_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        fixed_weights = torch.tensor(
            [[[0.1, 0.2, 0.3]] * int(4 / conv_layer_torch.groups)] * 8,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False
        conv_layer = tp.Conv(
            4,
            8,
            (3,),
            padding=test_case.tp_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=tp_dtype,
        )
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)
        if test_case.bias:
            conv_layer.bias = tp.cast(tp.Tensor(conv_layer_torch.bias.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = compile_fixture(conv_layer, input)

        # FP32 kernel seems to lose some precision, and FP16 needs to be run in FP32 on torch
        rtol_ = 4e-5 if tp_dtype == tp.float32 else 1e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    @pytest.mark.parametrize("test_case", test_cases_2d)
    def test_convolution_2d(self, torch_dtype, tp_dtype, test_case, compile_fixture):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1, 1)
        if not test_case.dilation:
            test_case.dilation = (1, 1)

        input_torch = torch.arange(200, dtype=torch.float32, device=torch.device("cuda")).reshape(*(2, 4, 5, 5))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv2d(
            4,
            8,
            3,
            padding=test_case.torch_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        fixed_weights = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]] * int(4 / (conv_layer_torch.groups))] * 8,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False
        conv_layer = tp.Conv(
            4,
            8,
            (3, 3),
            padding=test_case.tp_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=tp_dtype,
        )
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)
        if test_case.bias:
            conv_layer.bias = tp.cast(tp.Tensor(conv_layer_torch.bias.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = compile_fixture(conv_layer, input)

        rtol_ = 2e-7 if tp_dtype == tp.float32 else 1.5e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    @pytest.mark.parametrize("test_case", test_cases_3d)
    def test_convolution_3d(self, torch_dtype, tp_dtype, test_case, compile_fixture):
        pytest.skip("TODO (#260): Fix accuracy bugs in 3D conv")
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1, 1, 1)
        if not test_case.dilation:
            test_case.dilation = (1, 1, 1)

        input_torch = torch.arange(500, dtype=torch.float32, device=torch.device("cuda")).reshape(1, 4, 5, 5, 5) * 0.1
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv3d(
            4,
            8,
            3,
            padding=test_case.torch_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        fixed_weights = torch.tensor(
            [[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]] * 3] * int(4 / (conv_layer_torch.groups))] * 8,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.Conv(
            4,
            8,
            (3, 3, 3),
            padding=test_case.tp_pad,
            stride=test_case.stride,
            groups=test_case.groups,
            dilation=test_case.dilation,
            bias=test_case.bias,
            dtype=tp_dtype,
        )
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)
        if test_case.bias:
            conv_layer.bias = tp.cast(tp.Tensor(conv_layer_torch.bias.data), tp_dtype)

        # 3d grouped conv with fp16 is not supported by TRT
        if test_case.groups > 1 and tp_dtype == tp.float16:
            with helper.raises(
                tp.TripyException,
                has_stack_info_for=[input],
            ):
                output = conv_layer(input)
                output.eval()
            return

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = compile_fixture(conv_layer, input)

        rtol_ = 2e-4 if tp_dtype == tp.float32 else 1.4e-3  # 3d conv has greater accumulation error
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    def test_uneven_padding(self, torch_dtype, tp_dtype, compile_fixture):
        input_torch = torch.arange(200, dtype=torch.float32, device=torch.device("cuda")).reshape(*(2, 4, 5, 5))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        tp_pad = ((3, 1), (1, 2))
        torch_pad = torch.nn.ZeroPad2d((1, 2, 3, 1))  # only exposed for 2d in torch

        conv_layer_torch = torch.nn.Conv2d(
            4, 8, 3, padding=0, bias=False, dtype=torch.float32, device=torch.device("cuda")
        )
        fixed_weights = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]] * int(4 / (conv_layer_torch.groups))] * 8,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False
        conv_layer = tp.Conv(
            4,
            8,
            (3, 3),
            padding=tp_pad,
            bias=False,
            dtype=tp_dtype,
        )
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        input_torch = torch_pad(input_torch)
        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = compile_fixture(conv_layer, input)

        rtol_ = 2e-7 if tp_dtype == tp.float32 else 2e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape
