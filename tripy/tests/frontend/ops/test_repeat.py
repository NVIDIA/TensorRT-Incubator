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
from tests import helper
import nvtripy as tp


class TestRepeat:
    def test_invalid_dim_fails(self):
        a = tp.ones((2, 2))
        with helper.raises(tp.TripyException, "Dimension argument is out of bounds."):
            tp.repeat(a, 2, dim=4)

    def test_negative_repeats_fails(self):
        a = tp.ones((2, 2))
        with helper.raises(tp.TripyException, "`repeats` value must be non-negative."):
            tp.repeat(a, -1, dim=0)
