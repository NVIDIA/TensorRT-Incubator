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
from nvtripy.frontend.utils import tensor_from_shape_like


@pytest.mark.parametrize(
    "shape, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([tp.DimensionSize(1), tp.DimensionSize(2)], [1, 2]),
        ([], []),
        ([1, tp.DimensionSize(2), 3], [1, 2, 3]),
        ([1, tp.DimensionSize(2), 3, 4], [1, 2, 3, 4]),
    ],
)
def test_tensor_from_shape_like(shape, expected):
    tensor = tensor_from_shape_like(shape)

    assert tensor.tolist() == expected
