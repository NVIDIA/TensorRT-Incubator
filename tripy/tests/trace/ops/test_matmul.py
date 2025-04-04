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

import nvtripy as tp
import pytest


class TestMatMul:
    @pytest.mark.parametrize(
        "a, b, expected_rank",
        [
            (
                tp.ones((2,)),
                tp.ones((2,)),
                0,
            ),
            (tp.ones((2, 3)), tp.ones((3, 2)), 2),
            (tp.ones((4, 2, 3)), tp.ones((3, 2)), 3),
        ],
    )
    def test_infer_rank(self, a, b, expected_rank):
        out = a @ b
        assert out.trace_tensor.rank == expected_rank
