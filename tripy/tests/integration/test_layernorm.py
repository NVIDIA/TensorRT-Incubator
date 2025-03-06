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

import numpy as np
import re
import torch
import pytest

import nvtripy as tp
from tests import helper

DTYPES = [(torch.float16, tp.float16), (torch.float32, tp.float32)]


class TestLayerNorm:

    @pytest.mark.parametrize("torch_dtype, tp_dtype", DTYPES)
    @pytest.mark.parametrize("input_shape", [(2, 2, 2)])
    @pytest.mark.parametrize("normalized_shape", [(2, 2), (2,)])
    def test_layernorm_accuracy(self, torch_dtype, tp_dtype, input_shape, normalized_shape, eager_or_compiled):
        eps = 1e-5
        layernorm = torch.nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=torch_dtype,
            device="cuda",
        )
        tp_layernorm = tp.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            dtype=tp_dtype,
        )

        # use Tripy's parameters
        tp_layernorm.weight = tp.Tensor(layernorm.weight.detach())
        tp_layernorm.bias = tp.Tensor(layernorm.bias.detach())

        input = torch.arange(torch.prod(torch.Tensor(input_shape))).reshape(input_shape).to(torch_dtype).to("cuda")
        tp_input = tp.Tensor(input, dtype=tp_dtype)

        output = eager_or_compiled(tp_layernorm, tp_input)
        with torch.no_grad():
            expected = layernorm(input)

        rtol_ = 2e-7 if tp_dtype == tp.float32 else 1e-3
        assert torch.allclose(torch.from_dlpack(output), expected, rtol=rtol_)

    def test_layernorm_improper_dimensions(self):
        tp_layernorm = tp.LayerNorm(
            normalized_shape=[2, 2],
        )
        x = tp.ones((5, 5, 5))
        with helper.raises(
            tp.TripyException,
            match="broadcast dimensions must be conformable",
        ):
            tp_layernorm(x).eval()
