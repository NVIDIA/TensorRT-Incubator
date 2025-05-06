#
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

import numpy as np
import pytest
import nvtripy as tp


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [0, 1, None, [0, 1], [1, 0], -1, -2, [0, -1], [-2, 1]],
    )
    def test_flip(self, dims, eager_or_compiled):
        np_a = np.arange(16).reshape((4, 4)).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f = tp.flip(a, dim=dims)
        assert np.array_equal(np.from_dlpack(tp.copy(f, device=tp.device("cpu"))), np.flip(np_a.get(), axis=dims))

        # also ensure that flipping a second time restores the original value
        f2 = eager_or_compiled(tp.flip, f, dim=dims)
        assert np.array_equal(np.from_dlpack(tp.copy(f2, device=tp.device("cpu"))), np_a)

    def test_no_op(self, eager_or_compiled):
        np_a = np.arange(16).reshape((4, 4)).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f = eager_or_compiled(tp.flip, a, dim=[])
        assert tp.equal(a, f)

    def test_zero_rank(self, eager_or_compiled):
        t = tp.Tensor(1)
        f = eager_or_compiled(tp.flip, t)
        assert tp.equal(t, f)

    @pytest.mark.parametrize(
        "dims1, dims2",
        [(0, -2), (1, -1), ([0, 1], None), ([0, 1], [1, 0]), ([0, 1], [-2, -1])],
    )
    def test_equivalences(self, dims1, dims2, eager_or_compiled):
        np_a = np.arange(16).reshape((4, 4)).astype(np.float32)
        a = tp.Tensor(np_a, device=tp.device("gpu"))
        f1 = eager_or_compiled(tp.flip, a, dim=dims1)
        f2 = eager_or_compiled(tp.flip, a, dim=dims2)
        assert tp.equal(f1, f2)
