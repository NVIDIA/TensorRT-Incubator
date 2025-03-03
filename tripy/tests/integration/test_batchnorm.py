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
import torch
import pytest
import re

import nvtripy as tp
from tests import helper

DTYPES = [(torch.float32, tp.float32), (torch.float16, tp.float16)]


class TestBatchNorm:

    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(2, 2, 2, 2)])
    def test_batchnorm_accuracy(self, torch_dtype, tp_dtype, input_shape, eager_or_compiled):
        eps = 1e-5
        num_features = input_shape[1]  # Number of channels in the input tensor
        batchnorm = torch.nn.BatchNorm2d(num_features=num_features, eps=eps, dtype=torch_dtype)
        tp_batchnorm = tp.BatchNorm(
            num_features=num_features,
            dtype=tp_dtype,
            eps=eps,
        )

        # Use Tripy's parameters and ensure they match the dtype
        tp_batchnorm.weight = tp.Tensor(batchnorm.weight.detach())
        tp_batchnorm.bias = tp.Tensor(batchnorm.bias.detach())
        tp_batchnorm.running_mean = tp.Tensor(batchnorm.running_mean.detach())
        tp_batchnorm.running_var = tp.Tensor(batchnorm.running_var.detach())

        input = torch.randn(input_shape, dtype=torch_dtype).to("cuda")
        tp_input = tp.Tensor(input, dtype=tp_dtype)

        output = eager_or_compiled(tp_batchnorm, tp_input)

        batchnorm.to("cuda").eval()
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
            match=re.escape("broadcast dimensions must be conformable"),
        ):
            tp_batchnorm(x).eval()
