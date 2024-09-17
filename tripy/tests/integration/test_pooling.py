#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tripy as tp


class TestPooling:

    @pytest.mark.parametrize(
        "kernel_dims, stride, padding",
        [
            ((3, 3), (1, 1), ((0, 0), (0, 0))),
            ((4, 4), (2, 2), ((1, 1), (2, 2))),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16, tp.int8])
    def test_maxpool_2d(self, kernel_dims, stride, padding, dtype):
        inp_tp = tp.reshape(tp.arange(64, dtype=dtype), (1, 1, 8, 8))
        out = tp.maxpool(inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)

        torch_padding = (padding[0][0], padding[1][0])
        pool_torch = torch.nn.MaxPool2d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)
        expected = pool_torch(torch.from_dlpack(inp_tp).to("cpu"))
        assert torch.allclose(expected, torch.from_dlpack(out).to("cpu"))

    @pytest.mark.parametrize(
        "kernel_dims, stride, padding",
        [
            ((2, 2, 2), (2, 2, 2), ((0, 0), (1, 1), (1, 1))),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_maxpool_3d(self, kernel_dims, stride, padding, dtype):
        inp_tp = tp.reshape(tp.arange(512, dtype=dtype), (1, 1, 8, 8, 8))
        out = tp.maxpool(inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)

        torch_padding = (padding[0][0], padding[1][0], padding[2][0])
        pool_torch = torch.nn.MaxPool3d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)
        expected = pool_torch(torch.from_dlpack(inp_tp).to("cpu"))
        assert torch.allclose(expected, torch.from_dlpack(out).to("cpu"))
