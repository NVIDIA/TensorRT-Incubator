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
from nvtripy.frontend.utils import pretty_print


@pytest.mark.parametrize(
    "data_list, shape, expected_output",
    [
        ([1, 2, 3, 4, 5], [5], "[1, 2, 3, 4, 5]"),
        (list(range(100)), [100], "[0, 1, 2, ..., 97, 98, 99]"),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [3, 3], "[[1, 2, 3],\n [4, 5, 6],\n [7, 8, 9]]"),
        ([], [0], "[]"),
        ([1.1, 2.2, 3.3, 4.4, 5.5], [5], "[1.1, 2.2, 3.3, 4.4, 5.5]"),
        ([1, 2, 3, 4, 5], [5], "[1, 2, 3, 4, 5]"),
        ([True, False, True, False], [4], "[True, False, True, False]"),
    ],
)
def test_pretty_print(data_list, shape, expected_output):
    assert pretty_print(data_list, shape) == expected_output
