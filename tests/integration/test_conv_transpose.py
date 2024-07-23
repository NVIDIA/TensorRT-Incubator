from collections.abc import Sequence
from dataclasses import dataclass

import cupy as cp
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


test_cases_transpose_1d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((2, 2),), torch_pad=2),
    ConvTestCase(stride=(1,)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(2,)),
    ConvTestCase(bias=True),
]

test_cases_transpose_2d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((2, 2), (1, 1)), torch_pad=(2, 1)),
    ConvTestCase(stride=(1, 1)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(1, 2)),
    ConvTestCase(bias=True),
]

test_cases_transpose_3d = [
    ConvTestCase(),
    ConvTestCase(tp_pad=((2, 2), (1, 1), (2, 2)), torch_pad=(2, 1, 2)),
    ConvTestCase(stride=(1, 1, 1)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(3, 1, 2)),
    ConvTestCase(bias=True),
]

test_cases_transpose_downscale = [
    ConvTestCase(stride=(1, 1), tp_pad=((1, 1), (1, 1)), torch_pad=(1, 1)),
    ConvTestCase(stride=(2, 2), tp_pad=((2, 2), (2, 2)), torch_pad=(2, 2)),
]


# TODO (#147): Update tests to use Torch FP16 convolution


@pytest.mark.parametrize("torch_dtype,tp_dtype", DTYPES)
class TestConvolution:
    @pytest.mark.parametrize("test_case", test_cases_transpose_1d)
    def test_transposed_convolution_1d(self, torch_dtype, tp_dtype, test_case):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (2,)
        if not test_case.dilation:
            test_case.dilation = (1,)

        input_torch = torch.arange(12, dtype=torch.float32, device=torch.device("cuda")).reshape(*(1, 4, 3))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.ConvTranspose1d(
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
            [[[0.1, 0.2, 0.3]] * int(8 / (conv_layer_torch.groups))] * 4,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.ConvTranspose(
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
        output = conv_layer(input)

        rtol_ = 1e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    @pytest.mark.parametrize("test_case", test_cases_transpose_2d)
    def test_transposed_convolution_2d(self, torch_dtype, tp_dtype, test_case):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (2, 2)
        if not test_case.dilation:
            test_case.dilation = (1, 1)

        input_torch = torch.arange(36, dtype=torch.float32, device=torch.device("cuda")).reshape(*(1, 4, 3, 3))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.ConvTranspose2d(
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
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]] * int(8 / (conv_layer_torch.groups))] * 4,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.ConvTranspose(
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
        output = conv_layer(input)

        rtol_ = 1e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    @pytest.mark.parametrize("test_case", test_cases_transpose_3d)
    def test_transposed_convolution_3d(self, torch_dtype, tp_dtype, test_case):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (2, 2, 2)
        if not test_case.dilation:
            test_case.dilation = (1, 1, 1)

        input_torch = torch.arange(108, dtype=torch.float32, device=torch.device("cuda")).reshape(*(1, 4, 3, 3, 3))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.ConvTranspose3d(
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
            [[[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]] * 3] * int(8 / (conv_layer_torch.groups))] * 4,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.ConvTranspose(
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

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)
        rtol_ = 1.3e-6 if tp_dtype == tp.float32 else 1.6e-3
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape

    def test_transposed_equivalency(self, torch_dtype, tp_dtype):
        input_torch = torch.arange(9, dtype=torch.float32, device=torch.device("cuda")).reshape(*(1, 1, 3, 3))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv2d(
            1, 1, 3, padding=2, bias=False, dtype=torch.float32, device=torch.device("cuda")
        )

        fixed_weights = torch.tensor(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]]],
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights

        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.Conv(1, 1, (3, 3), padding=((2, 2), (2, 2)), bias=False, dtype=tp_dtype)
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        conv_transpose_layer_torch = torch.nn.ConvTranspose2d(
            1, 1, 3, bias=False, dtype=torch.float32, device=torch.device("cuda")
        )
        conv_transpose_layer_torch.weight.data = torch.flip(conv_layer_torch.weight.data, [2, 3])
        for param in conv_transpose_layer_torch.parameters():
            param.requires_grad = False

        conv_transpose_layer = tp.ConvTranspose(1, 1, (3, 3), stride=(1, 1), bias=False, dtype=tp_dtype)
        conv_transpose_layer.weight = tp.cast(tp.Tensor(conv_transpose_layer_torch.weight.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        expected_transpose = conv_transpose_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)
        output_transpose = conv_transpose_layer(input)

        rtol_ = 2e-7 if tp_dtype == tp.float32 else 9e-4
        output_torch = torch.from_dlpack(output)
        output_transpose_torch = torch.from_dlpack(output_transpose)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape
        assert torch.allclose(output_transpose_torch, expected_transpose, rtol=rtol_)
        assert output_transpose_torch.shape == expected_transpose.shape
        assert torch.allclose(output_torch, output_transpose_torch, rtol=rtol_)
        assert output_torch.shape == output_transpose_torch.shape
        assert torch.allclose(expected, expected_transpose, rtol=rtol_)
        assert expected.shape == expected_transpose.shape

    @pytest.mark.parametrize("test_case", test_cases_transpose_downscale)
    def test_transposed_downscale(self, torch_dtype, tp_dtype, test_case):
        input_torch = torch.arange(9, dtype=torch.float32, device=torch.device("cuda")).reshape(*(1, 1, 3, 3))
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.ConvTranspose2d(
            1,
            1,
            1,
            stride=test_case.stride,
            padding=test_case.torch_pad,
            bias=False,
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        fixed_weights = torch.tensor(
            [[[[0.1]]]],
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        conv_layer_torch.weight.data = fixed_weights
        for param in conv_layer_torch.parameters():
            param.requires_grad = False

        conv_layer = tp.ConvTranspose(
            1, 1, (1, 1), stride=test_case.stride, padding=test_case.tp_pad, bias=False, dtype=tp_dtype
        )
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)

        rtol_ = 1e-15 if tp_dtype == tp.float32 else 1e-10
        output_torch = torch.from_dlpack(output)
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape
