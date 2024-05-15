import pytest
from tests import helper

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import tripy as tp

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
    ConvTestCase(tp_pad=((0, 0), (3, 3), (0, 0)), torch_pad=(0, 3, 0)),
    ConvTestCase(stride=(3, 1, 3)),
    ConvTestCase(groups=2),
    ConvTestCase(groups=4),
    ConvTestCase(dilation=(2, 2, 2)),
    ConvTestCase(bias=True),
]


# TODO (#147): Update tests to use Torch FP16 convolution
@pytest.mark.parametrize("torch_dtype,tp_dtype", DTYPES)
class TestConvolution:
    @pytest.mark.parametrize("test_case", test_cases_1d)
    def test_convolution_1d(self, torch_dtype, tp_dtype, test_case):
        # TODO (#138): Switch to tripy random for tests
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1,)
        if not test_case.dilation:
            test_case.dilation = (1,)

        input_torch = torch.randn((2, 4, 5), dtype=torch.float32)
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
        )
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
        output = conv_layer(input)

        # FP32 kernel seems to lose some precision, and FP16 needs to be run in FP32 on torch
        atol_ = 1e-3 if tp_dtype == tp.float32 else 5e-3
        # TODO(#188): Remove cast when dlpack supports all precisions
        assert torch.allclose(torch.from_numpy(output.numpy()).to(torch_dtype), expected, atol=atol_)

    @pytest.mark.parametrize("test_case", test_cases_2d)
    def test_convolution_2d(self, torch_dtype, tp_dtype, test_case):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1, 1)
        if not test_case.dilation:
            test_case.dilation = (1, 1)

        input_torch = torch.randn((2, 4, 5, 5), dtype=torch.float32)
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
        )
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
        output = conv_layer(input)

        atol_ = 1e-3 if tp_dtype == tp.float32 else 5e-3
        # TODO(#188): Remove cast when dlpack supports all precisions
        assert torch.allclose(torch.from_numpy(output.numpy()).to(torch_dtype), expected, atol=atol_)

    @pytest.mark.parametrize("test_case", test_cases_3d)
    def test_convolution_3d(self, torch_dtype, tp_dtype, test_case):
        if not test_case.torch_pad:
            test_case.torch_pad = 0
        if not test_case.stride:
            test_case.stride = (1, 1, 1)
        if not test_case.dilation:
            test_case.dilation = (1, 1, 1)

        input_torch = torch.randn((2, 4, 5, 5, 5), dtype=torch.float32)
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
        )
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
        output = conv_layer(input)

        atol_ = 1e-3 if tp_dtype == tp.float32 else 2e-2  # 3d conv has greater accumulation error
        # TODO(#188): Remove cast when dlpack supports all precisions
        assert torch.allclose(torch.from_numpy(output.numpy()).to(torch_dtype), expected, atol=atol_)

    def test_uneven_padding(self, torch_dtype, tp_dtype):
        input_torch = torch.randn((2, 4, 5, 5), dtype=torch.float32)
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        tp_pad = ((3, 1), (1, 2))
        torch_pad = torch.nn.ZeroPad2d((1, 2, 3, 1))  # only exposed for 2d in torch

        conv_layer_torch = torch.nn.Conv2d(4, 8, 3, padding=0, bias=False, dtype=torch.float32)
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
        output = conv_layer(input)

        atol_ = 1e-3 if tp_dtype == tp.float32 else 5e-3
        # TODO(#188): Remove cast when dlpack supports all precisions
        assert torch.allclose(torch.from_numpy(output.numpy()).to(torch_dtype), expected, atol=atol_)
