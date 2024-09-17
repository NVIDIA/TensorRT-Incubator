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
from tripy.frontend.module.parameter import DefaultParameter
from tripy.frontend.trace.ops import Storage
import pytest


class TestParameter:
    def test_is_instance_of_tensor(self):
        param = tp.Parameter(tp.Tensor([1, 2, 3]))
        assert isinstance(param, tp.Parameter)

        tensor = tp.Tensor([1, 2, 3])
        assert not isinstance(tensor, tp.Parameter)

    def test_is_equivalent_to_tensor(self):
        tensor = tp.Tensor([1, 2, 3])
        param = tp.Parameter(tensor)

        assert np.array_equal(cp.from_dlpack(param).get(), cp.from_dlpack(tensor).get())

    def test_can_construct_from_non_tensor(self):
        param = tp.Parameter([1, 2, 3])
        assert np.array_equal(cp.from_dlpack(param).get(), np.array([1, 2, 3], dtype=np.int32))

    @pytest.mark.parametrize(
        "other,is_compatible",
        [
            (tp.Parameter(tp.ones((1, 2), dtype=tp.float32)), True),
            # Different shape
            (tp.Parameter(tp.ones((2, 2), dtype=tp.float32)), False),
            # Different dtype
            (tp.Parameter(tp.ones((1, 2), dtype=tp.float16)), False),
        ],
    )
    def test_is_compatible(self, other, is_compatible):
        param = tp.Parameter(tp.ones((1, 2), dtype=tp.float32))

        assert bool(param._is_compatible(other)) == is_compatible


class TestDefaultParameter:
    @pytest.mark.parametrize(
        "other,is_compatible",
        [
            (tp.Parameter(tp.ones((1, 2), dtype=tp.float32)), True),
            # Different shape
            (tp.Parameter(tp.ones((2, 2), dtype=tp.float32)), False),
            # Different dtype
            (tp.Parameter(tp.ones((1, 2), dtype=tp.float16)), False),
        ],
    )
    def test_is_compatible(self, other, is_compatible):
        param = DefaultParameter((1, 2), dtype=tp.float32)

        assert bool(param._is_compatible(other)) == is_compatible

    def test_is_compatible_does_not_materialize_data(self):
        param = DefaultParameter((1, 2), dtype=tp.float32)
        other = tp.Parameter(tp.ones((1, 2), dtype=tp.float32))

        assert param._is_compatible(other)
        assert not isinstance(param.trace_tensor.producer, Storage)

    def test_data_can_be_materialized(self):
        param = DefaultParameter((1, 2), dtype=tp.float32)
        assert np.array_equal(cp.from_dlpack(param).get(), np.array([[0, 1]], dtype=np.float32))
