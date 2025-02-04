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

import nvtripy as tp


class TestBinaryElementwise:
    @pytest.mark.parametrize(
        "lhs, rhs, expected_rank",
        [
            (tp.Tensor([1.0]), tp.Tensor([2.0]), 1),
            (tp.Tensor([1.0]), 2.0, 1),
            (1.0, tp.Tensor([2.0]), 1),
            (tp.ones((2, 3)), 2.0, 2),
        ],
    )
    def test_infer_rank(self, lhs, rhs, expected_rank):
        out = lhs + rhs
        assert out.trace_tensor.rank == expected_rank
