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
from tripy.frontend.trace.ops import UnaryElementwise


_UNARY_OPS = {
    UnaryElementwise.Kind.EXP: tp.exp,
    UnaryElementwise.Kind.TANH: tp.tanh,
    UnaryElementwise.Kind.RSQRT: tp.rsqrt,
    UnaryElementwise.Kind.LOG: tp.log,
    UnaryElementwise.Kind.SQRT: tp.sqrt,
    UnaryElementwise.Kind.ABS: tp.abs,
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _UNARY_OPS.items()])
    def test_op_funcs(self, func, kind):
        a = tp.Tensor([1.0])

        out = func(a)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, UnaryElementwise)
        assert out.trace_tensor.producer.kind == kind

    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _UNARY_OPS.items()])
    def test_infer_rank(self, func, kind):
        a = tp.ones((2, 3))
        out = func(a)
        assert out.trace_tensor.rank == 2
