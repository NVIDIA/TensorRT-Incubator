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
import pytest

import tripy as tp


class TestCopy:
    @pytest.mark.parametrize("src", ["cpu", "gpu"])
    @pytest.mark.parametrize("dst", ["cpu", "gpu"])
    def test_single_copy(self, src, dst):
        a = tp.Tensor([1, 2], device=tp.device(src))
        out = tp.copy(a, tp.device(dst))
        assert out.tolist() == [1, 2]
        assert out.device.kind == dst

    def test_multiple_copy_1(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("gpu"))
        out = tp.copy(a, tp.device("cpu"))
        assert out.tolist() == [1, 2]
        assert out.device.kind == "cpu"

    def test_multiple_copy_2(self):
        a = tp.Tensor([1, 2])
        a = tp.copy(a, tp.device("cpu"))
        out = tp.copy(a, tp.device("gpu"))
        assert out.tolist() == [1, 2]
        assert out.device.kind == "gpu"
