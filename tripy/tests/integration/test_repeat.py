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

import pytest
import numpy as np


class TestRepeat:
    @pytest.mark.parametrize(
        "repeats,dim",
        [
            (1, 0),
            (2, 0),
            (2, -1),
            (2, 1),
            (0, 1),
        ],
    )
    def test_repeat(self, repeats, dim):
        inp = np.arange(4, dtype=np.int32).reshape((2, 2))

        out = tp.repeat(tp.Tensor(inp), repeats, dim)
        expected = np.repeat(inp, repeats, dim)

        assert np.array_equal(np.from_dlpack(tp.copy(out, device=tp.device("cpu"))), expected)
