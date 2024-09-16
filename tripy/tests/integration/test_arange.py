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

import cupy as cp
import numpy as np
import tripy as tp
from tests import helper
import pytest


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

    def test_tensor_inputs(self):
        start = tp.Tensor(1.0)
        stop = tp.Tensor(5.0)
        step = tp.Tensor(0.5)
        out = tp.arange(start, stop, step)
        assert np.allclose(cp.from_dlpack(out).get(), np.arange(1.0, 5.0, 0.5, dtype=np.float32))
