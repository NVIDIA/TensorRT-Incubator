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

import tripy as tp

from tripy.frontend.trace.ops import Fill


class TestFull:
    def test_op_func(self):
        a = tp.full([1, 2], 1, dtype=tp.int32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Fill)

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
