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


class TestUnsqueezeOp:
    @pytest.mark.parametrize("axis", [-1, 0, 2])
    def test_unsqueeze_dynamic_op(self, axis):
        def func(a):
            return tp.unsqueeze(a, dim=axis)

        inp = np.ones((4, 2, 2, 3), dtype=np.float32)

        out = func(tp.Tensor(inp))
        ref_out = np.expand_dims(inp, axis=axis)
        assert tp.allclose(out, tp.Tensor(ref_out))

        assert out.shape == ref_out.shape
