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

import nvtripy as tp


class TestPooling:

    @pytest.mark.parametrize(
        "kernel_dims, stride, padding",
        [
            ((3, 3), (1, 1), ((0, 0), (0, 0))),
            ((4, 4), (2, 2), ((1, 1), (2, 2))),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16, tp.int8])
    @pytest.mark.parametrize("pool_type", ["max", "avg"])
    def test_pool_2d(self, kernel_dims, stride, padding, dtype, pool_type, eager_or_compiled):
        inp_tp = tp.reshape(tp.arange(64, dtype=dtype), (1, 1, 8, 8))
        torch_padding = (padding[0][0], padding[1][0])

        if pool_type == "avg" and dtype == tp.int8:
            pytest.skip("Torch average pool is not implemented for int8")

        if pool_type == "max":
            out = eager_or_compiled(tp.maxpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)
            pool_torch = torch.nn.MaxPool2d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)
        elif pool_type == "avg":
            if torch_padding != (0, 0):
                pytest.skip(
                    "https://github.com/NVIDIA/TensorRT-Incubator/issues/241: Tripy average pool is incorrect when padding != 0."
                )

            out = eager_or_compiled(tp.avgpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)
            pool_torch = torch.nn.AvgPool2d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)

        out_torch = torch.from_dlpack(out).to("cpu")
        expected = pool_torch(torch.from_dlpack(inp_tp).to("cpu"))
        assert torch.allclose(expected, out_torch)
        assert expected.shape == out_torch.shape

    @pytest.mark.parametrize(
        "kernel_dims, stride, padding",
        [
            ((2, 2, 2), (2, 2, 2), ((0, 0), (1, 1), (1, 1))),
        ],
    )
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    @pytest.mark.parametrize("pool_type", ["max", "avg"])
    def test_pool_3d(self, kernel_dims, stride, padding, dtype, pool_type, eager_or_compiled):
        inp_tp = tp.reshape(tp.arange(512, dtype=dtype), (1, 1, 8, 8, 8))
        torch_padding = (padding[0][0], padding[1][0], padding[2][0])

        if torch_padding != (0, 0, 0):
            pytest.skip(
                "https://github.com/NVIDIA/TensorRT-Incubator/issues/241: Tripy average pool is incorrect when padding != 0."
            )

        if pool_type == "max":
            out = eager_or_compiled(tp.maxpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)
            pool_torch = torch.nn.MaxPool3d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)
        elif pool_type == "avg":
            out = eager_or_compiled(tp.avgpool, inp_tp, kernel_dims=kernel_dims, stride=stride, padding=padding)
            pool_torch = torch.nn.AvgPool3d(kernel_size=kernel_dims, stride=stride, padding=torch_padding)

        out_torch = torch.from_dlpack(out).to("cpu")

        expected = pool_torch(torch.from_dlpack(inp_tp).to("cpu"))
        assert torch.allclose(expected, out_torch)
        assert expected.shape == out_torch.shape
