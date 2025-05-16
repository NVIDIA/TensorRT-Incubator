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

import pytest
import torch

import nvtripy as tp

DTYPES = [(torch.float16, tp.float16), (torch.float32, tp.float32)]


class TestGroupNorm:

    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(1, 10, 2)])
    @pytest.mark.parametrize("num_groups", [2, 5])
    @pytest.mark.parametrize("num_channels", [10])
    def test_groupnorm_accuracy(self, torch_dtype, tp_dtype, input_shape, num_groups, num_channels, eager_or_compiled):
        eps = 1e-5
        groupnorm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            dtype=torch_dtype,
            device="cuda",
        )
        tp_groupnorm = tp.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            dtype=tp_dtype,
        )

        # default is 1s and 0s which will only test an edge case of the affine transform
        torch.nn.init.uniform_(groupnorm.weight)
        torch.nn.init.uniform_(groupnorm.bias)

        tp_groupnorm.weight = tp.Tensor(groupnorm.weight.to("cpu").detach())
        tp_groupnorm.bias = tp.Tensor(groupnorm.bias.to("cpu").detach())

        input = torch.arange(torch.prod(torch.Tensor(input_shape))).reshape(input_shape).to(torch_dtype).to("cuda")
        # input = input / 100.0 + 0.5
        tp_input = tp.Tensor(input, dtype=tp_dtype)

        output = eager_or_compiled(tp_groupnorm, tp_input)
        with torch.no_grad():
            expected = groupnorm(input)

        print("expected:", expected)
        print("output:", output)

        diff = torch.from_dlpack(output) - expected

        # Maximum absolute difference
        max_abs_diff = torch.max(torch.abs(diff))
        print(f"Maximum absolute difference: {max_abs_diff}\n")

        # Maximum relative difference
        # Add small epsilon to denominator to avoid division by 0
        eps = 1e-8
        rel_diff = torch.abs(diff) / (torch.abs(expected) + eps)
        max_rel_diff = torch.max(rel_diff)
        print(f"Maximum relative difference: {max_rel_diff}\n")

        rtol_ = 1e-04 if tp_dtype == tp.float32 else 1e-1
        assert tp.allclose(output, tp.Tensor(expected), rtol=rtol_)
