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

DTYPES = [(torch.float16, tp.float16), (torch.float32, tp.float32)]
INPUT_SHAPES = [
    (2, 3, 4),
    (1, 5, 8, 8),
    (1, 4, 3, 6, 6),
]


class TestInstanceNorm:

    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", INPUT_SHAPES)
    def test_instancenorm_accuracy(self, torch_dtype, tp_dtype, input_shape, eager_or_compiled):
        eps = 1e-5
        num_channels = input_shape[1]

        # Choose appropriate PyTorch InstanceNorm based on input dimensions
        TorchInstanceNorm = {3: torch.nn.InstanceNorm1d, 4: torch.nn.InstanceNorm2d, 5: torch.nn.InstanceNorm3d}[
            len(input_shape)
        ]
        instancenorm = TorchInstanceNorm(num_features=num_channels, eps=eps, affine=True).to(torch_dtype).to("cuda")

        tp_instancenorm = tp.InstanceNorm(
            num_channels=num_channels,
            eps=eps,
            dtype=tp_dtype,
        )

        torch.nn.init.uniform_(instancenorm.weight)
        torch.nn.init.uniform_(instancenorm.bias)

        tp_instancenorm.weight = tp.Tensor(instancenorm.weight.to("cpu").detach())
        tp_instancenorm.bias = tp.Tensor(instancenorm.bias.to("cpu").detach())

        input = torch.arange(torch.prod(torch.Tensor(input_shape))).reshape(input_shape).to(torch_dtype).to("cuda")
        input = input / 100.0 + 0.5
        tp_input = tp.Tensor(input, dtype=tp_dtype)

        output = eager_or_compiled(tp_instancenorm, tp_input)
        with torch.no_grad():
            expected = instancenorm(input)

        rtol_ = 1e-4 if tp_dtype == tp.float32 else 1e-2
        atol_ = 1e-4 if tp_dtype == tp.float32 else 1e-2

        torch_output = torch.from_dlpack(output)

        assert torch_output.shape == expected.shape

        assert torch.allclose(torch_output, expected, rtol=rtol_, atol=atol_)
