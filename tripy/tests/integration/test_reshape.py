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
import cupy as cp
import pytest
import nvtripy as tp


class TestReshape:
    @pytest.mark.parametrize(
        "shape, new_shape",
        [
            ((2, 4), (1, 8)),
            ((2, 4, 8, 9), (8, 8, 9)),
            ((2, 4), (8,)),  # change rank of output
            ((2, 4), (1, -1)),  # check negative dim
        ],
    )
    def test_static_reshape(self, shape, new_shape, eager_or_compiled):
        cp_a = cp.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        a = tp.Tensor(cp_a)
        b = eager_or_compiled(tp.reshape, a, new_shape)
        if -1 in new_shape:
            new_shape = tuple(np.prod(shape) // -np.prod(new_shape) if d == -1 else d for d in new_shape)
        assert np.array_equal(cp.from_dlpack(b).get(), cp_a.reshape(new_shape).get())

    def test_reshape_shape_tensor(self, eager_or_compiled):
        a = tp.ones((2, 3, 4))
        b = tp.ones((2, 3, 2, 2))
        out = eager_or_compiled(tp.reshape, a, (a.shape[0], a.shape[1], b.shape[2], b.shape[3]))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 3, 2, 2), dtype=np.float32))

    def test_reshape_shape_with_unknown(self, eager_or_compiled):
        a = tp.ones((2, 3, 4))
        out = eager_or_compiled(tp.reshape, a, (2, a.shape[1], a.shape[2] / 2, -1))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 3, 2, 2), dtype=np.float32))
