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


class TestResize:

    @pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
    def test_scales(self, mode):
        inp_torch = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
        inp_tp = tp.Tensor(inp_torch)
        out_tp = tp.resize(inp_tp, mode, scales=(1, 1, 2, 2))
        torch_mode = {
            "nearest": "nearest",
            "linear": "bilinear",
            "cubic": "bicubic",
        }[mode]
        expected = torch.nn.functional.interpolate(inp_torch, scale_factor=2, mode=torch_mode)
        out_torch = torch.from_dlpack(out_tp).to("cpu")
        assert list(expected.shape) == out_tp.shape
        assert torch.allclose(out_torch, expected)

    @pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
    def test_output_shape(self, mode):
        inp_torch = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
        inp_tp = tp.Tensor(inp_torch)
        out_tp = tp.resize(inp_tp, mode, output_shape=[1, 1, 8, 8])
        torch_mode = {
            "nearest": "nearest",
            "linear": "bilinear",
            "cubic": "bicubic",
        }[mode]
        expected = torch.nn.functional.interpolate(inp_torch, size=[8, 8], mode=torch_mode)
        out_torch = torch.from_dlpack(out_tp).to("cpu")
        assert list(expected.shape) == out_tp.shape
        assert torch.allclose(out_torch, expected)
