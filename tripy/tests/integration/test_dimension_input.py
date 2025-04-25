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

import nvtripy as tp
import cupy as cp
import pytest


@pytest.fixture(scope="session")
def dim_input_executable():
    def func(x, y):
        x = x + x
        x = tp.reshape(x, (-1, y))
        return x

    return tp.compile(
        func, args=[tp.InputInfo(shape=((2, 4, 6), 4), dtype=tp.float32), tp.DimensionInputInfo(value_bounds=(1, 2, 3))]
    )


class TestDimensionInput:
    def test_dimension_input(self, dim_input_executable):
        inp_cp = cp.arange(16, dtype=cp.float32).reshape((4, 4))
        inp = tp.Tensor(inp_cp)
        dim_inp = tp.DimensionSize(2)
        out = dim_input_executable(inp, dim_inp)
        expected = (inp_cp + inp_cp).reshape((-1, 2))
        assert cp.array_equal(cp.from_dlpack(out), expected)
