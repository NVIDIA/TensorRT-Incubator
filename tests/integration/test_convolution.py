import pytest

import torch
import tripy as tp

DTYPES = [
    (torch.float16, tp.float16),
    (torch.float32, tp.float32),
]


# TODO (#147): Update tests to use Torch FP16 convolution
@pytest.mark.parametrize("torch_dtype,tp_dtype", DTYPES)
class TestConvolution:
    def test_convolution_1d(self, torch_dtype, tp_dtype):
        # TODO (#138): Switch to tripy random for tests
        input_torch = torch.randn((4, 3, 8), dtype=torch.float32)
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv1d(3, 16, 5, bias=False, dtype=torch.float32)
        conv_layer_torch.requires_grad = False
        conv_layer = tp.Conv(3, 16, (5,), dtype=tp_dtype)
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)

        # FP32 kernel seems to lose some precision, and FP16 needs to be run in FP32 on torch
        atol_ = 1e-3 if tp_dtype == tp.float32 else 5e-3
        assert torch.allclose(torch.from_numpy(output.numpy()), expected, atol=atol_)

    def test_convolution_2d(self, torch_dtype, tp_dtype):
        input_torch = torch.randn((4, 3, 8, 8), dtype=torch.float32)
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv2d(3, 16, 5, bias=False, dtype=torch.float32)
        conv_layer_torch.requires_grad = False
        conv_layer = tp.Conv(3, 16, (5, 5), dtype=tp_dtype)
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)

        atol_ = 1e-3 if tp_dtype == tp.float32 else 5e-3
        assert torch.allclose(torch.from_numpy(output.numpy()), expected, atol=atol_)

    def test_convolution_3d(self, torch_dtype, tp_dtype):
        input_torch = torch.randn((4, 3, 8, 8, 8), dtype=torch.float32)
        input = tp.cast(tp.Tensor(input_torch), tp_dtype)

        conv_layer_torch = torch.nn.Conv3d(3, 16, 5, bias=False, dtype=torch.float32)
        conv_layer_torch.requires_grad = False
        conv_layer = tp.Conv(3, 16, (5, 5, 5), dtype=tp_dtype)
        conv_layer.weight = tp.cast(tp.Tensor(conv_layer_torch.weight.data), tp_dtype)

        expected = conv_layer_torch(input_torch).to(torch_dtype)
        output = conv_layer(input)

        atol_ = 1e-3 if tp_dtype == tp.float32 else 1e-2  # 3d conv has greater accumulation error
        assert torch.allclose(torch.from_numpy(output.numpy()), expected, atol=atol_)
