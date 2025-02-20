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
import cupy as cp
import numpy as np

import nvtripy as tp


class TestPad:

    @pytest.mark.parametrize(
        "pad, value",
        [
            (((0, 1), (2, 0)), 0),
            (((1, 2), (2, 3)), 1),
        ],
    )
    def test_pad_constant(self, pad, value, eager_or_compiled):
        inp = np.arange(4, dtype=np.int32).reshape((2, 2))

        out = eager_or_compiled(tp.pad, tp.Tensor(inp), pad, value=value)
        expected = np.pad(inp, pad, constant_values=value)

        assert np.array_equal(cp.from_dlpack(out).get(), expected)

    def test_pad_tensor(self, eager_or_compiled):
        inp = np.arange(6, dtype=np.float32).reshape((2, 3))

        inp_tp = tp.Tensor(inp)
        out = eager_or_compiled(tp.pad, tp.Tensor(inp), ((0, inp_tp.shape[0]), (inp_tp.shape[1], 0)))
        expected = np.pad(inp, ((0, 2), (3, 0)))

        assert np.array_equal(cp.from_dlpack(out).get(), expected)
