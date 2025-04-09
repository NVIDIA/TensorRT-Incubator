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

from tests import helper
import nvtripy as tp


class TestPooling:
    def test_invalid_kernel_dims(self):
        a = tp.ones((1, 1, 4, 4))
        with helper.raises(tp.TripyException, "Unsupported kernel_dims, must be 2D or 3D."):
            tp.maxpool(a, (2,))

    def test_invalid_stride(self):
        a = tp.ones((1, 1, 4, 4))
        with helper.raises(tp.TripyException, "Stride must have the same length as kernel_dims."):
            tp.maxpool(a, (2, 2), stride=(1,))

    def test_invalid_padding_length(self):
        a = tp.ones((1, 1, 4, 4))
        with helper.raises(tp.TripyException, "Padding must have the same length as kernel_dims."):
            tp.maxpool(a, (2, 2), padding=((1, 1),))

    def test_invalid_padding_contents(self):
        a = tp.ones((1, 1, 4, 4))
        with helper.raises(
            tp.TripyException,
            "For parameter: 'padding', expected an instance of type: 'Sequence[Tuple[int, int]] | None'",
        ):
            tp.maxpool(a, (2, 2), padding=((1, 1, 1), (1, 1, 1)))
