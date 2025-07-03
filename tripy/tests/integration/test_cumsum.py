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
import pytest

import nvtripy as tp


class TestCumsum:
    @pytest.mark.parametrize(
        "data,dim,expected",
        [
            ([0, 1, 2, 3], 0, [0, 1, 3, 6]),
            # Negative dim:
            ([[2, 3], [4, 5]], -2, [[2, 3], [6, 8]]),
            # Non-innermost dim:
            ([[2, 3], [4, 5]], 0, [[2, 3], [6, 8]]),
            # >2D (can potentially find transposition bugs)
            ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 0, [[[1, 2], [3, 4]], [[6, 8], [10, 12]]]),
        ],
    )
    def test_cumsum(self, data, dim, expected, eager_or_compiled):
        inp = tp.cast(tp.Tensor(data), tp.float32)

        out = eager_or_compiled(tp.cumsum, inp, dim=dim)
        expected = tp.cast(tp.Tensor(expected), tp.float32)
        assert tp.allclose(out, expected)
        assert out.shape == expected.shape
