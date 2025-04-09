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
from tests import helper
import nvtripy as tp
import pytest


class TestUnsqueeze:
    @pytest.mark.parametrize("dim", [-4, 3])
    def test_out_of_bounds_dimension(self, dim):
        x = tp.ones((2, 3))
        with helper.raises(tp.TripyException, match="Dimension argument is out of bounds"):
            tp.unsqueeze(x, dim=dim)

    @pytest.mark.parametrize("dim, expected_shape", [(-3, (1, 2, 3)), (2, (2, 3, 1))])
    def test_in_bounds_dimension(self, dim, expected_shape):
        # Unsqueeze inserts a dimension *before* the specified dimension, meaning
        # it can be "out of bounds" by 1 and still be valid.
        x = tp.ones((2, 3))
        y = tp.unsqueeze(x, dim=dim)
        assert y.shape == expected_shape
