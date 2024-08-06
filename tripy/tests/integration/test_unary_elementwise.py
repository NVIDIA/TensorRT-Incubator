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
import pytest

import tripy as tp

_UNARY_OPS = {
    tp.exp: np.exp,
    tp.tanh: np.tanh,
    tp.rsqrt: lambda a: 1.0 / np.sqrt(a),
    tp.log: np.log,
    tp.sin: np.sin,
    tp.cos: np.cos,
    tp.sqrt: np.sqrt,
    tp.abs: np.abs,
}


class TestUnaryElementwise:
    @pytest.mark.parametrize("tp_func, np_func", [(tp_func, np_func) for tp_func, np_func in _UNARY_OPS.items()])
    def test_op_funcs(self, tp_func, np_func, compile_fixture):
        input = tp.arange(1, 4, dtype=tp.float32)
        output = compile_fixture(tp_func, input)
        assert tp.allclose(output, tp.Tensor(np_func(cp.from_dlpack(input).get())))
