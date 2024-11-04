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

import cupy as cp

import tripy as tp


class TestModulo:
    def test_mod(self):
        input = tp.arange(1, 4, dtype=tp.float32)
        output = input % 2
        assert tp.allclose(output, tp.Tensor(cp.from_dlpack(input).get() % 2))

    def test_rmod(self):
        input = tp.arange(1, 4, dtype=tp.float32)
        output = 2 % input
        assert tp.allclose(output, tp.Tensor(2 % cp.from_dlpack(input).get()))
