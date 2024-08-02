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

import pytest

import tripy as tp

from tripy.common.datatype import DATA_TYPES
from tripy.frontend.trace.ops import Fill, Cast


class TestFull:
    def test_op_func(self):
        a = tp.full([1, 2], 1, dtype=tp.int32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    @pytest.mark.parametrize("value, fill_no_cast_dtype", [(True, tp.bool), (1.0, tp.float32), (1, tp.int32)])
    def test_explicit_cast(self, dtype, value, fill_no_cast_dtype):
        if dtype == tp.int4:
            pytest.skip(f"Unsupported front-end data type {dtype}")
        if dtype == tp.float8:
            pytest.skip(f"Data can not be implicitly converted to {dtype}")

        a = tp.full([1, 2], value, dtype=dtype)

        if dtype == fill_no_cast_dtype:
            assert isinstance(a.trace_tensor.producer, Fill)
        else:
            assert isinstance(a.trace_tensor.producer, Cast)
            assert isinstance(a.trace_tensor.producer.inputs[0].producer, Fill)

    def test_infer_rank(self):
        a = tp.full((2, 3), 1)
        assert a.trace_tensor.rank == 2

    def test_shape_is_shape_tensor(self):
        shape = tp.ones((2, 3)).shape
        a = tp.full(shape, 1, dtype=tp.int32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)
        assert a.trace_tensor.rank == 2

    def test_scalar_convert_to_shape_tensor(self):
        shape = tp.ones((2, 3)).shape
        a = tp.full((shape[0],), 1, dtype=tp.int32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)
        assert a.trace_tensor.rank == 1


class TestFullLike:
    def test_op_func(self):
        t = tp.Tensor([[1, 2], [3, 4]])
        a = tp.full_like(t, 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)

    def test_infer_rank(self):
        t = tp.ones((3, 5, 1))
        a = tp.full_like(t, 2)
        assert a.trace_tensor.rank == 3
