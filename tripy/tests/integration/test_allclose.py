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


class TestAllClose:
    @pytest.mark.parametrize(
        "tensor_a, tensor_b, rtol, atol, expected",
        [
            ([1e10, 1e-7], [1e10, 1e-8], 1e-05, 1e-08, False),
            ([1e10, 1e-8], [1.0001e10, 1e-9], 1e-05, 1e-08, False),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.0, 0.01, True),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.01, 0.0, True),
            ([1.0, 2.0, 3.0], [1.01, 2.01, 3.01], 0.01, 0.01, True),
        ],
    )
    def test_all_close_float32(self, tensor_a, tensor_b, rtol, atol, expected):
        out = tp.allclose(
            tp.Tensor(tensor_a, dtype=tp.float32), tp.Tensor(tensor_b, dtype=tp.float32), rtol=rtol, atol=atol
        )
        assert out == expected
