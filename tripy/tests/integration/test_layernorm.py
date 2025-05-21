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


class TestLayerNorm:

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(2, 8, 4, 4)])
    @pytest.mark.parametrize(
        "normalized_shape",
        [
            (8, 4, 4),
            (4, 4),
        ],
    )
    def test_layernorm_accuracy(self, dtype, input_shape, normalized_shape, eager_or_compiled):
        eps = 0.0
        torch_dtype = TORCH_DTYPES[dtype]
        layernorm = torch.nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=torch_dtype,
            device="cuda",
        )
        tp_layernorm = tp.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=dtype,
        )

        input = torch.empty(*input_shape, dtype=torch_dtype, device="cuda").uniform_(0, 10)
        tp_input = tp.Tensor(input, dtype=dtype)

        # Verify normalized output has approximately mean=0, std=1
        tp_layernorm.weight = tp.ones(normalized_shape, dtype=dtype)
        tp_layernorm.bias = tp.zeros(normalized_shape, dtype=dtype)

        output = eager_or_compiled(tp_layernorm, tp_input)

        output_torch = torch.from_dlpack(output)

        normalized_dims = list(range(len(input_shape) - len(normalized_shape), len(input_shape)))
        mean = output_torch.mean(dim=tuple(normalized_dims), keepdim=True)
        var = output_torch.var(dim=tuple(normalized_dims), keepdim=True, unbiased=False)

        mean_abs = mean.abs().mean().item()
        var_diff = (var - 1).abs().mean().item()

        assert mean_abs < 1e-4, f"Mean should be close to 0, got {mean_abs}"
        assert var_diff < 1e-4, f"Variance should be close to 1, got {var_diff}"

        # Comparison test with the affine transformation included
        torch.nn.init.uniform_(layernorm.weight, 0.2, 2)
        torch.nn.init.uniform_(layernorm.bias, 0.2, 2)

        tp_layernorm.weight = tp.Tensor(layernorm.weight.to("cpu").detach())
        tp_layernorm.bias = tp.Tensor(layernorm.bias.to("cpu").detach())

        output = eager_or_compiled(tp_layernorm, tp_input)
        with torch.no_grad():
            expected = layernorm(input)

        atol_ = 1e-6 if dtype == tp.float32 else 5e-3

        torch_output = torch.from_dlpack(output)
        assert torch_output.shape == expected.shape
        assert torch.allclose(torch_output, expected, atol=atol_)
