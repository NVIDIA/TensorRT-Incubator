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
from tripy.common.exception import TripyException

DTYPES = [
    (torch.float16, tp.float16),
    (torch.float32, tp.float32)
]

class TestGroupNorm:

    @pytest.mark.l1
    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(1, 10, 2)])
    @pytest.mark.parametrize("num_groups", [2, 5])
    @pytest.mark.parametrize("num_channels", [10])
    def test_groupnorm_accuracy(self, torch_dtype, tp_dtype, input_shape, num_groups, num_channels):
        eps = 1e-5
        groupnorm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            dtype=torch_dtype,
        )
        tp_groupnorm = tp.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            dtype=tp_dtype,
        )

        tp_groupnorm.weight = tp.Parameter(groupnorm.weight.detach())
        tp_groupnorm.bias = tp.Parameter(groupnorm.bias.detach())

        input = torch.arange(torch.prod(torch.Tensor(input_shape))).reshape(input_shape).to(torch_dtype)
        tp_input = tp.Tensor(input, dtype=tp_dtype)

        output = tp.copy(tp_groupnorm(tp_input), tp.device("cpu"))
        with torch.no_grad():
            expected = groupnorm(input)

        rtol_ = 2e-6 if tp_dtype == tp.float32 else 1e-3
        assert torch.allclose(torch.from_dlpack(output), expected, rtol=rtol_)
