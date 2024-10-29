import torch
import pytest
import re

import tripy as tp
from tests import helper

DTYPES = [(torch.float32, tp.float32), (torch.float16, tp.float16)]

class TestBatchNorm:

    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(2, 2, 2, 2)])
    def test_batchnorm_accuracy(self, torch_dtype, tp_dtype, input_shape):
        eps = 1e-5
        num_features = input_shape[1]  # Number of channels in the input tensor
        batchnorm = torch.nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            dtype=torch_dtype
        )
        tp_batchnorm = tp.BatchNorm(
            num_features=num_features,
            dtype=tp_dtype,
            eps=eps,
        )
    
        # Use Tripy's parameters and ensure they match the dtype
        tp_batchnorm.weight = tp.Parameter(batchnorm.weight.detach())
        tp_batchnorm.bias = tp.Parameter(batchnorm.bias.detach())
        tp_batchnorm.running_mean = tp.Parameter(batchnorm.running_mean.detach())
        tp_batchnorm.running_var = tp.Parameter(batchnorm.running_var.detach())

    
        input = torch.randn(input_shape, dtype=torch_dtype).to('cuda')
        tp_input = tp.Tensor(input, dtype=tp_dtype)
    
        output = tp_batchnorm(tp_input)

        batchnorm.to('cuda').eval()
        with torch.no_grad():
            expected = batchnorm(input)

        rtol_ = 2e-7 if tp_dtype == tp.float32 else 1e-3
        assert torch.allclose(torch.from_dlpack(output), expected, rtol=rtol_)


    def test_batchnorm_improper_dimensions(self):
        num_features = 2
        tp_batchnorm = tp.BatchNorm(
            num_features=num_features,
        )
        x = tp.ones((3, 3, 3)) 
        with helper.raises(
            tp.TripyException,
            match=re.escape("size of operand dimension 1 (2) is not compatible with size of result dimension 1 (3)"),
        ):
            tp_batchnorm(x).eval()
