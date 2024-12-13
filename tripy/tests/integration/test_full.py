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
import numpy as np

import tripy as tp


class TestFull:
    def test_normal_shape(self, eager_or_compiled):
        out = eager_or_compiled(tp.full, (2, 2), 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 2), 5.0, np.float32))

    def test_shape_tensor(self, eager_or_compiled):
        a = tp.ones((2, 3))
        out = eager_or_compiled(tp.full, a.shape, 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 3), 5.0, np.float32))

    def test_mixed_shape(self, eager_or_compiled):
        a = tp.ones((2, 3))
        out = eager_or_compiled(tp.full, (a.shape[0], 4), 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 4), 5.0, np.float32))

    def test_value_as_tensor(self, eager_or_compiled):
        a = tp.ones((2, 3))
        out = eager_or_compiled(tp.full, (a.shape[0], 4), tp.Tensor(8.0), tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 4), 8.0, np.float32))
