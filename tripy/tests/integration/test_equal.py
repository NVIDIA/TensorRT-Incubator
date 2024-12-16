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
import pytest

import nvtripy as tp


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (tp.Tensor([1, 2], dtype=tp.float32), tp.Tensor([1, 2], dtype=tp.float32), True),
        (tp.ones((2, 2), dtype=tp.int32), tp.Tensor([[1, 1], [1, 1]], dtype=tp.int32), True),
        (tp.ones((1, 4)), tp.ones((4, 1)), False),
    ],
)
def test_equal(a, b, expected):
    out = tp.equal(a, b)
    assert out == expected
