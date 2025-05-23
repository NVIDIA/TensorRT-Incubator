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

import nvtripy as tp
import pytest
import torch

from tests.helper import TORCH_DTYPES

DTYPES = [tp.float16, tp.float32]
INPUT_SHAPES = [
    (2, 3, 4),
    (1, 5, 8, 8),
    (1, 4, 3, 6, 6),
]

dtype_params = pytest.mark.parametrize("dtype", DTYPES)
input_shape_params = pytest.mark.parametrize("input_shape", INPUT_SHAPES)


@pytest.fixture
def setup(dtype, input_shape):
    torch_dtype = TORCH_DTYPES[dtype]
    eps = 0.0
    num_channels = input_shape[1]

    # Choose appropriate PyTorch InstanceNorm based on input dimensions
    TorchInstanceNorm = {3: torch.nn.InstanceNorm1d, 4: torch.nn.InstanceNorm2d, 5: torch.nn.InstanceNorm3d}[
        len(input_shape)
    ]
    instancenorm = TorchInstanceNorm(num_features=num_channels, eps=eps, affine=True).to(torch_dtype).to("cuda")

    tp_instancenorm = tp.InstanceNorm(
        num_channels=num_channels,
        eps=eps,
        dtype=dtype,
    )

    input = torch.empty(*input_shape, dtype=torch_dtype, device="cuda").uniform_(0, 10)
    tp_input = tp.Tensor(input, dtype=dtype)

    yield instancenorm, tp_instancenorm, tp_input, num_channels


class TestInstanceNorm:

    @dtype_params
    @input_shape_params
    def test_instancenorm_normalization(self, input_shape, setup, eager_or_compiled):
        """Test that normalized output has approximately mean=0, std=1"""
        _, tp_instancenorm, tp_input, num_channels = setup
        dtype = tp_instancenorm.weight.dtype

        tp_instancenorm.weight = tp.ones((num_channels,), dtype=dtype)
        tp_instancenorm.bias = tp.zeros((num_channels,), dtype=dtype)

        output = eager_or_compiled(tp_instancenorm, tp_input)
        output_torch = torch.from_dlpack(output)

        spatial_dims = tuple(range(2, len(input_shape)))
        means = output_torch.mean(dim=spatial_dims, keepdim=True)
        vars = output_torch.var(dim=spatial_dims, keepdim=True, unbiased=False)

        mean_abs = means.abs().mean().item()
        var_diff = (vars - 1).abs().mean().item()

        assert mean_abs < 2e-4, f"Instance mean should be close to 0, got {mean_abs}"
        assert var_diff < 1e-3, f"Instance variance should be close to 1, got {var_diff}"

    @dtype_params
    @input_shape_params
    def test_instancenorm_affine_transformation(self, setup, eager_or_compiled):
        """Test the InstanceNorm with affine transformation included"""
        instancenorm, tp_instancenorm, tp_input, _ = setup
        dtype = tp_instancenorm.weight.dtype
        input = torch.from_dlpack(tp_input)

        torch.nn.init.uniform_(instancenorm.weight, 0.2, 2)
        torch.nn.init.uniform_(instancenorm.bias, 0.2, 2)

        tp_instancenorm.weight = tp.Tensor(instancenorm.weight.to("cpu").detach())
        tp_instancenorm.bias = tp.Tensor(instancenorm.bias.to("cpu").detach())

        output = eager_or_compiled(tp_instancenorm, tp_input)
        with torch.no_grad():
            expected = instancenorm(input)

        atol_ = 1e-6 if dtype == tp.float32 else 6e-3

        torch_output = torch.from_dlpack(output)
        assert torch_output.shape == expected.shape
        assert torch.allclose(torch_output, expected, atol=atol_)
