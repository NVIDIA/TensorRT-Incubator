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
from tests import helper

import nvtripy as tp


class TestInput:
    @pytest.mark.parametrize(
        "shape, expected_min, expected_opt, expected_max",
        [
            # int:
            # min/opt/max explicitly specified
            ([(1, 2, 3)], [1], [2], [3]),
            # Only one value specified
            ([1], [1], [1], [1]),
            # `DimensionSize`s:
            # min/opt/max explicitly specified
            ([(tp.DimensionSize(1), tp.DimensionSize(2), tp.DimensionSize(3))], [1], [2], [3]),
            # Only one value specified
            ([tp.DimensionSize(1)], [1], [1], [1]),
        ],
    )
    def test_shapes_normalized(self, shape, expected_min, expected_opt, expected_max):
        inp = tp.InputInfo(shape=shape, dtype=tp.float32)

        assert inp.shape_bounds.min == expected_min
        assert inp.shape_bounds.opt == expected_opt
        assert inp.shape_bounds.max == expected_max

    @pytest.mark.parametrize(
        "shape",
        [
            # Not a number
            (tp.int32, 1),
            # Too few elements in dimension
            ((1, 1), 1),
            # Too many elements in dimension
            ((1, 1, 1, 1), 1),
            # Tuple containing a non-number
            ((tp.int32, 1, 1), 1),
        ],
    )
    def test_invalid_shape(self, shape):
        with helper.raises(
            tp.TripyException,
            r"Not a valid overload because: For parameter: 'shape', expected an instance of type: 'Sequence\[int \| nvtripy.DimensionSize | Tuple\[int \| nvtripy.DimensionSize, int \| nvtripy.DimensionSize, int \| nvtripy.DimensionSize\]\]' but got argument of type: ",
        ):
            tp.InputInfo(shape=shape, dtype=tp.float32)
