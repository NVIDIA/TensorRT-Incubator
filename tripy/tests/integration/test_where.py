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

import cupy as cp
import numpy as np
import pytest

import nvtripy as tp
from nvtripy.frontend import Tensor


class TestWhereOp:
    @pytest.mark.parametrize(
        "cond, x, y",
        [
            ((1), (2, 3), (2, 3)),  # Broadcast condition
            ((2), (2, 2), (2, 2)),  # Add extra batch
            ((1,), (1, 3), (1, 3)),
            ((2, 2), (1,), (2,)),  # Broadcast x and y (not equal)
            ((1,), (0,), (1,)),  # Broadcast with a 0 dim
            ((0,), (1,), (1,)),  # 0 dim in the condition
        ],
    )
    def test_broadcast_shapes(self, cond, x, y, eager_or_compiled):
        x = np.arange(np.prod(x)).reshape(x).astype(np.float32)
        y = np.arange(np.prod(y)).reshape(y).astype(np.float32)
        t_cond = np.arange(np.prod(cond)).reshape(cond).astype(np.float32)
        a = Tensor(x)
        b = Tensor(y)
        condition = Tensor(t_cond % 2 == 0)
        out = eager_or_compiled(tp.where, condition, a, b)
        assert np.array_equal(cp.from_dlpack(out).get(), np.array(np.where((t_cond % 2 == 0), x, y)))

    def test_dimension_size_inputs(self):
        condition = tp.Tensor(True)
        out = tp.where(condition, tp.DimensionSize(0), tp.DimensionSize(1))
        assert isinstance(out, tp.DimensionSize)
        assert out.tolist() == 0

    def test_explicit_condition(self, eager_or_compiled):
        select_indices = tp.Tensor([True, False, True, False])
        ones = tp.ones((4,), dtype=tp.int32)
        zeros = tp.zeros((4,), dtype=tp.int32)
        w = eager_or_compiled(tp.where, select_indices, ones, zeros)
        assert cp.from_dlpack(w).get().tolist() == [1, 0, 1, 0]
