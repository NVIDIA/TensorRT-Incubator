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
import pytest
from tests import helper


class TestSlice:
    def test_incorrect_index_size(self):
        with helper.raises(
            tp.TripyException,
            match=r"Input tensor has a rank of 2 but was attempted to be sliced with 3 indices",
        ):
            a = tp.Tensor([[1, 2], [3, 4]])
            b = a[:, :, 0:1]
            b.eval()

    def test_invalid_index(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(tp.TripyException, match="out of bounds access"):
            a[3].eval()

    def test_invalid_multiple_dims(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(tp.TripyException, match="out of bounds access"):
            a[5, 3].eval()

    @pytest.mark.parametrize(
        "start",
        [
            "hi",
            tp.Tensor(0),
        ],
    )
    def test_invalid_slice_type(self, start):
        a = tp.ones((4,))
        with helper.raises(tp.TripyException, match="Slice start must be an integer or a DimensionSize."):
            a[start:].eval()
