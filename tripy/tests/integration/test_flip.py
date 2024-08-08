#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import pytest
import tripy as tp


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [0, 1, None, [0, 1], [1, 0], -1, -2, [0, -1], [-2, 1]],
    )
    def test_flip(self, dims, compile_fixture):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f = compile_fixture(tp.flip, a, dims=dims)
        assert np.array_equal(cp.from_dlpack(f).get(), np.flip(cp_a.get(), axis=dims))

        # also ensure that flipping a second time restores the original value
        f2 = compile_fixture(tp.flip, f, dims=dims)
        assert cp.array_equal(cp.from_dlpack(f2), cp_a)

    def test_no_op(self):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f = tp.flip(a, dims=[])
        assert cp.array_equal(cp.from_dlpack(a), cp.from_dlpack(f))

    def test_zero_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert cp.array_equal(cp.from_dlpack(t), cp.from_dlpack(f))

    @pytest.mark.parametrize(
        "dims1, dims2",
        [(0, -2), (1, -1), ([0, 1], None), ([0, 1], [1, 0]), ([0, 1], [-2, -1])],
    )
    def test_equivalences(self, dims1, dims2):
        cp_a = cp.arange(16).reshape((4, 4)).astype(cp.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        f1 = tp.flip(a, dims=dims1)
        f2 = tp.flip(a, dims=dims2)
        assert cp.array_equal(cp.from_dlpack(f1), cp.from_dlpack(f2))
