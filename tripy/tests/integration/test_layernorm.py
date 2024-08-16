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

import numpy as np
import torch
import pytest

import tripy as tp

DTYPES = [
    (torch.float16, tp.float16),
    (torch.float32, tp.float32),
]

@pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
class TestLayerNorm:
    @pytest.mark.parametrize("input_shape", [(10, 10, 5)])
    @pytest.mark.parametrize("normalized_shape", [(10, 5), (5,)])
    @pytest.mark.parametrize("eps", [1e-5, 1e-3])
    def test_layernorm_module(self, torch_dtype, tp_dtype, device, input_shape, normalized_shape, eps):
        torch_device = "cuda" if device == "gpu" else device
        tp_layernorm = tp.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=tp_dtype,
        )
        layernorm = torch.nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=torch_dtype,
            device=torch_device
        )

        # use Tripy's parameters
        layernorm.weight = torch.nn.Parameter(torch.from_dlpack(tp_layernorm.weight.__dlpack__()).to(torch_device))
        layernorm.bias = torch.nn.Parameter(torch.from_dlpack(tp_layernorm.bias.__dlpack__()).to(torch_device))

        input = torch.arange(torch.prod(torch.Tensor(input_shape))).reshape(input_shape).to(torch_device).to(torch_dtype)
        tp_input = tp.Tensor(input, dtype=tp_dtype, device=tp.device(device))

        output = tp_layernorm(tp_input)
        expected = layernorm(input).to("cpu")

        output_torch = torch.from_dlpack(output).to("cpu")
        rtol_ = 2e-7 if tp_dtype == tp.float32 else 1.5e-3
        assert torch.allclose(output_torch, expected, rtol=rtol_)
        assert output_torch.shape == expected.shape