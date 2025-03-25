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
import math


class TestPooling:
    @pytest.mark.parametrize(
        "kernel_dims, stride, padding",
        [
            ((3, 3), (1, 1), ((0, 0), (0, 0))),
            ((4, 4), (2, 2), ((1, 1), (2, 2))),
            ((2, 2, 2), (2, 2, 2), ((0, 0), (1, 1), (1, 1))),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    @pytest.mark.parametrize("pool_type", ["max", "avg"])
    def test_pool(self, kernel_dims, stride, padding, dtype, pool_type, eager_or_compiled):
        input_shape = (1, 1) + (8,) * len(kernel_dims)
        inp_tp = tp.reshape(tp.arange(math.prod(input_shape), dtype=dtype), input_shape)
        torch_padding = []
        # Torch padding starts from the last dimension
        for tup in reversed(padding):
            torch_padding.extend(tup)

        if pool_type == "max":
            out = eager_or_compiled(tp.maxpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)

            TorchPoolType = torch.nn.MaxPool2d if len(kernel_dims) == 2 else torch.nn.MaxPool3d
            pool_torch = TorchPoolType(kernel_size=kernel_dims, stride=stride)
        elif pool_type == "avg":
            out = eager_or_compiled(tp.avgpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)

            TorchPoolType = torch.nn.AvgPool2d if len(kernel_dims) == 2 else torch.nn.AvgPool3d
            pool_torch = TorchPoolType(kernel_size=kernel_dims, stride=stride)

        out_torch = torch.from_dlpack(out)

        padded_inp = torch.nn.functional.pad(torch.from_dlpack(inp_tp), torch_padding)
        expected = pool_torch(padded_inp)
        assert torch.allclose(expected, out_torch)
        assert expected.shape == out_torch.shape
