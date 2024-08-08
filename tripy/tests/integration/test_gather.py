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


class TestGatherOp:
    @pytest.mark.parametrize(
        "x_shape, axis, indices",
        [
            ((2, 3), 0, (1)),
            pytest.param((2, 3, 4), 1, (1, 2),marks=pytest.mark.xfail(reason="Bug in gather op. (Remove xfail when fixed)")),
            ((2, 3), 1, (1)),
            pytest.param((2, 3, 4), 0, (0, 1),marks=pytest.mark.xfail(reason="Bug in gather op. (Remove xfail when fixed)")),
            pytest.param((2, 3, 4), 1, (0, 1),marks=pytest.mark.xfail(reason="Bug in gather op. (Remove xfail when fixed)")),
            ((2, 3, 4), 1 , (2)),
        ],
    )
    def test_gather(self, x_shape, axis, indices):
        x = np.arange(np.prod(x_shape)).reshape(x_shape)
        indices_tp = tp.Tensor(indices)
        a = tp.Tensor(x)
        out = tp.gather(a, axis, indices_tp)
        out.eval()
        assert np.array_equal(cp.from_dlpack(out).get(), np.take(x, indices,axis))