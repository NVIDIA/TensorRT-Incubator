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

import nvtripy as tp
import pytest

_UNARY_OPS = [
    tp.exp,
    tp.tanh,
    tp.rsqrt,
    tp.log,
    tp.sqrt,
    tp.abs,
    tp.sigmoid,
    tp.sin,
    tp.cos,
]


class TestUnaryElementwise:
    @pytest.mark.parametrize("func", _UNARY_OPS)
    def test_infer_rank(self, func):
        a = tp.ones((2, 3))
        out = func(a)
        assert out.trace_tensor.rank == 2
