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

import cupy as cp
import numpy as np
import nvtripy as tp
import pytest
from tests import helper


class TestArange:
    def test_basic_functionality(self):
        out = tp.arange(0, 5)
        assert np.array_equal(cp.from_dlpack(out).get(), np.arange(0, 5, dtype=np.float32))

    def test_with_step(self):
        out = tp.arange(1, 10, 2)
        assert np.array_equal(cp.from_dlpack(out).get(), np.arange(1, 10, 2, dtype=np.float32))

    def test_negative_step(self):
        out = tp.arange(5, 0, -1)
        assert np.array_equal(cp.from_dlpack(out).get(), np.arange(5, 0, -1, dtype=np.float32))

    def test_float_values(self):
        out = tp.arange(0.5, 5.5, 0.5)
        assert np.allclose(cp.from_dlpack(out).get(), np.arange(0.5, 5.5, 0.5, dtype=np.float32))

    def test_single_parameter(self):
        out = tp.arange(5)
        assert np.array_equal(cp.from_dlpack(out).get(), np.arange(5, dtype=np.float32))

    def test_errors(self):
        with helper.raises(
            tp.TripyException,
            match="Step in arange cannot be 0.",
        ):
            tp.arange(0, 5, 0)

    def test_shapescalar_inputs(self):
        a = tp.ones((1, 5, 1))
        out = tp.arange(a.shape[0], a.shape[1] + 1, a.shape[2])
        assert np.allclose(cp.from_dlpack(out).get(), np.arange(1.0, 6.0, 1.0, dtype=np.float32))

    @pytest.mark.parametrize(
        "start,stop,step,expected",
        [
            (tp.DimensionSize(2), 5, 1, np.arange(2, 5, 1)),
            (tp.DimensionSize(2), tp.DimensionSize(8), tp.DimensionSize(3), np.arange(2, 8, 3)),
        ],
    )
    def test_dimensionsize_combinations(self, start, stop, step, expected):
        out = tp.arange(start, stop, step)
        assert np.allclose(cp.from_dlpack(out).get(), expected.astype(np.float32))
