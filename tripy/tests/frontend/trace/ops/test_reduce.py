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

import pytest

import tripy as tp
from tripy.frontend.trace.ops import Reduce, BinaryElementwise, ArgMinMax


class TestReduce:
    def test_sum(self):
        a = tp.ones((2, 3))
        a = tp.sum(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_max(self):
        a = tp.ones((2, 3))
        a = tp.max(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_all(self):
        a = tp.ones((2, 3))
        a = tp.all(a)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_any(self):
        a = tp.ones((2, 3))
        a = tp.any(a)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Reduce)

    def test_mean(self):
        a = tp.ones((2, 3))
        a = tp.mean(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, BinaryElementwise)
        assert a.trace_tensor.producer.kind == BinaryElementwise.Kind.DIV

    def test_variance(self):
        a = tp.ones((2, 3))
        a = tp.var(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, BinaryElementwise)
        assert a.trace_tensor.producer.kind == BinaryElementwise.Kind.DIV

    def test_argmax(self):
        a = tp.ones((2, 3))
        a = tp.argmax(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, ArgMinMax)

    def test_argmin(self):
        a = tp.ones((2, 3))
        a = tp.argmin(a, 0)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, ArgMinMax)

    @pytest.mark.parametrize(
        "func, expected_rank",
        [
            (lambda t: tp.sum(t, 0), 1),
            (lambda t: tp.sum(t), 0),
            (lambda t: tp.mean(t, 0), 1),
            (lambda t: tp.mean(t), 0),
        ],
    )
    def test_infer_rank(self, func, expected_rank):
        a = tp.ones((2, 3))
        out = func(a)
        assert out.trace_tensor.rank == expected_rank
