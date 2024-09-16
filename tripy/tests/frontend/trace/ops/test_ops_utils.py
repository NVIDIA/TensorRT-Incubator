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

from tripy.frontend.trace.ops.utils import get_broadcast_in_dim


class TestGetBroadcastInDim:
    @pytest.mark.parametrize(
        "input_shape, output_shape, expected_dim",
        [
            ([2, 2, 3], [2, 2, 3], [0, 1, 2]),  # no broadcast
            ([2, 3], [2, 2, 3], [1, 2]),  # simple broadcast
            ([], [2, 2, 3], []),  # output should be of same rank as input
            ([5], [2, 4, 5], [2]),
        ],
    )
    def test_static_broadcast_in_dim(self, input_shape, output_shape, expected_dim):
        assert get_broadcast_in_dim(len(input_shape), len(output_shape)) == expected_dim
