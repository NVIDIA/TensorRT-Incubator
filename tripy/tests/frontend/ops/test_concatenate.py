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
from tests import helper


class TestConcatenate:
    def test_mismatched_ranks(self):
        a = tp.ones((2, 3))
        b = tp.ones((2,))

        with helper.raises(tp.TripyException, match="Concatenated tensors must have equal ranks"):
            tp.concatenate([a, b], dim=0)

    @pytest.mark.parametrize("dim", [-3, 2])
    def test_out_of_bounds_dim(self, dim):
        a = tp.ones((2, 3))
        b = tp.ones((2, 3))

        with helper.raises(tp.TripyException, match="Dimension argument is out of bounds"):
            tp.concatenate([a, b], dim=dim)
