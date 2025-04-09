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

import re

import nvtripy as tp
from nvtripy.trace.ops.where import Where
from tests import helper


class TestWhere:
    def test_bool_condition(self):
        cond = tp.Tensor([False, True, False], dtype=tp.bool)
        a = tp.Tensor([1, 2, 3], dtype=tp.int32)
        b = tp.Tensor([4, 5, 6], dtype=tp.int32)
        w = tp.where(cond, a, b)
        assert isinstance(w, tp.Tensor)
        assert isinstance(w.trace_tensor.producer, Where)

    def test_mismatched_input_shapes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with helper.raises(tp.TripyException, match="broadcast dimensions must be conformable"):
            c.eval()

    def test_mismatched_input_dtypes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)

        with helper.raises(tp.TripyException, match="Mismatched data types in 'where'."):
            c = tp.where(cond, a, b)

    def test_condition_is_not_bool(self):
        cond = tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)

        with helper.raises(tp.TripyException, match="Unsupported data type in 'where'."):
            c = tp.where(cond, a, b)


class TestMaskedFill:
    def test_condition_is_not_bool(self):
        a = tp.Tensor([0, 1, 0, 1])
        mask = tp.Tensor([1.0, 2.0, 3.0, 4.0])

        with helper.raises(tp.TripyException, match="Unsupported data type in 'masked_fill'."):
            b = tp.masked_fill(a, mask, -1)
