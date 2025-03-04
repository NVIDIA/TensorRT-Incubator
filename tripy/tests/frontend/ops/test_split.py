#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tests import helper


class TestSplit:
    def test_empty_indices(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Split indices must not be empty"):
            sp = tp.split(t, [])

    def test_zero_splits(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"`num_split_or_sizes` must be positive"):
            sp = tp.split(t, 0)

    def test_invalid_split_dimension(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Dimension argument is out of bounds"):
            sp = tp.split(t, 5, dim=2)
