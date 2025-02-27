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
import nvtripy as tp

import pytest
import cupy as cp


class TestRepeat:
    @pytest.mark.parametrize(
        "repeats,dim",
        [
            (1, 0),
            (2, 0),
            (2, -1),
            (2, 1),
            (0, 1),
        ],
    )
    def test_repeat(self, repeats, dim, eager_or_compiled):
        inp = cp.arange(4, dtype=cp.int32).reshape((2, 2))

        out = eager_or_compiled(tp.repeat, tp.Tensor(inp), repeats, dim)
        expected = cp.repeat(inp, repeats, dim)

        assert cp.array_equal(cp.from_dlpack(out), expected)

    def test_repeat_shape_scalar(self, eager_or_compiled):
        inp = cp.arange(4, dtype=cp.int32).reshape((2, 2))
        s = tp.ones((1, 2))
        out = eager_or_compiled(tp.repeat, tp.Tensor(inp), repeats=s.shape[1], dim=0)
        expected = cp.repeat(inp, 2, 0)

        assert cp.array_equal(cp.from_dlpack(out), expected)
