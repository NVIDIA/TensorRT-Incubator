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


class TestResize:
    def test_unsupported_mode(self):
        with helper.raises(tp.TripyException, match="Unsupported resize mode."):
            a = tp.ones((1, 1, 3, 4))
            out = tp.resize(a, mode="bilinear", output_shape=(1, 1, 6, 8))

    def test_invalid_align_corners(self):
        with helper.raises(tp.TripyException, match="align_corners can only be set with `cubic` or `linear` mode."):
            a = tp.ones((1, 1, 3, 4))
            out = tp.resize(a, mode="nearest", output_shape=(1, 1, 6, 8), align_corners=True)
