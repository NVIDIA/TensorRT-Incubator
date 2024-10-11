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

import tripy as tp


class TestInput:
    @pytest.mark.parametrize(
        "shape, expected_min, expected_opt, expected_max",
        [
            # min/opt/max explicitly specified
            ([(1, 2, 3)], (1,), (2,), (3,)),
            # Only one value specified
            ([1], (1,), (1,), (1,)),
        ],
    )
    def test_shapes_normalized(self, shape, expected_min, expected_opt, expected_max):
        inp = tp.InputInfo(shape=shape, dtype=tp.float32)

        assert inp.shape_bounds.min == expected_min
        assert inp.shape_bounds.opt == expected_opt
        assert inp.shape_bounds.max == expected_max

    @pytest.mark.parametrize(
        "shape, expected_error",
        [
            # Not a number
            (
                (tp.int32, 1),
                "Shape values should be either a single number or a Tuple specifying min/opt/max bounds.",
            ),
            # Too few elements in dimension
            (((1, 1), 1), "Incorrect number of shape values provided"),
            # Too many elements in dimension
            (((1, 1, 1, 1), 1), "Incorrect number of shape values provided"),
            # Tuple containing a non-number
            (((tp.int32, 1, 1), 1), "Shape values must be numbers"),
        ],
    )
    def test_invalid_shape(self, shape, expected_error):
        with helper.raises(tp.TripyException, expected_error):
            tp.InputInfo(shape=shape, dtype=tp.float32)
