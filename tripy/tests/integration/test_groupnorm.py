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


class TestGroupNorm:

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(1, 6, 2, 2)])
    @pytest.mark.parametrize("num_groups", [2, 3])
    @pytest.mark.parametrize("num_channels", [6])
    def test_groupnorm_accuracy(self, dtype, input_shape, num_groups, num_channels, eager_or_compiled):
        eps = 0.0
        torch_dtype = TORCH_DTYPES[dtype]
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
            dtype=dtype,
        )

        input = torch.empty(*input_shape, dtype=torch_dtype, device="cuda").uniform_(0, 10)
        tp_input = tp.Tensor(input, dtype=dtype)

        # Verify normalized output has approximately mean=0, std=1
        tp_groupnorm.weight = tp.ones((num_channels,), dtype=dtype)
        tp_groupnorm.bias = tp.zeros((num_channels,), dtype=dtype)

        output = eager_or_compiled(tp_groupnorm, tp_input)
        output_torch = torch.from_dlpack(output)

        N, C = input_shape[0], input_shape[1]
        spatial_size = torch.prod(torch.tensor(input_shape[2:]))
        reshaped = output_torch.view(N, num_groups, C // num_groups, spatial_size)

        means = reshaped.mean(dim=(2, 3))
        vars = reshaped.var(dim=(2, 3), unbiased=False)

        mean_abs = means.abs().mean().item()
        var_diff = (vars - 1).abs().mean().item()

        assert mean_abs < 2e-4, f"Group mean should be close to 0, got {mean_abs}"
        assert var_diff < 1e-3, f"Group variance should be close to 1, got {var_diff}"

        # Comparison test with the affine transformation included
        torch.nn.init.uniform_(groupnorm.weight, 0.2, 2)
        torch.nn.init.uniform_(groupnorm.bias, 0.2, 2)

        tp_groupnorm.weight = tp.Tensor(groupnorm.weight.to("cpu").detach())
        tp_groupnorm.bias = tp.Tensor(groupnorm.bias.to("cpu").detach())

        output = eager_or_compiled(tp_groupnorm, tp_input)
        with torch.no_grad():
            expected = groupnorm(input)

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

        atol_ = 1e-6 if dtype == tp.float32 else 5e-3

        torch_output = torch.from_dlpack(output)
        assert torch_output.shape == expected.shape
        assert torch.allclose(torch_output, expected, atol=atol_)
