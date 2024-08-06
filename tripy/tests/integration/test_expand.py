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

import tripy as tp


class TestExpand:
    def test_int_sizes(self, compile_fixture):
        input = tp.ones((2, 1))
        out = compile_fixture(tp.expand, input, (-1, 2))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 2), dtype=np.float32))

    def test_shape_sizes(self, compile_fixture):
        input = tp.ones((2, 1))
        a = tp.ones((2, 4))
        out = compile_fixture(tp.expand, input, a.shape)
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 4), dtype=np.float32))

    def test_extra_dims(self, compile_fixture):
        input = tp.ones((2, 1))
        out = compile_fixture(tp.expand, input, (1, -1, 2))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((1, 2, 2), dtype=np.float32))

    def test_mixed_sizes(self, compile_fixture):
        input = tp.ones((2, 1, 1))
        a = tp.ones((4, 4))
        out = compile_fixture(tp.expand, input, (-1, a.shape[0], a.shape[1]))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 4, 4), dtype=np.float32))
