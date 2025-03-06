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


class TestExpand:
    def test_invalid_small_size(self):
        a = tp.ones((2, 1, 1))

        with helper.raises(
            tp.TripyException,
            match="The length of `sizes` must be greater or equal to input tensor's rank.",
        ):
            b = tp.expand(a, (2, 2))

    def test_invalid_prepended_dim(self):
        # We cannot use -1 if we are prepending a new dimension.
        a = tp.ones((2,))

        with helper.raises(tp.TripyException, match="Cannot use -1 for prepended dimension."):
            b = tp.expand(a, (-1, 2))

    def test_invalid_mismatch_size(self):
        a = tp.ones((2, 1))
        b = tp.expand(a, (4, 2))

        with helper.raises(tp.TripyException, match=r"broadcast dimensions must be conformable"):
            b.eval()
