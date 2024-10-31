#
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

import tripy as tp
from tripy.frontend.trace.ops import Resize
from tests import helper


class TestResize:
    def test_shape(self):
        a = tp.ones((1, 1, 3, 4))
        out = tp.resize(a, "nearest", output_shape=(1, 1, 6, 8))
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, Resize)

    def test_unsupported_mode(self):
        with helper.raises(tp.TripyException, match="Unsupported resize mode."):
            a = tp.ones((1, 1, 3, 4))
            out = tp.resize(a, "bilinear", output_shape=(1, 1, 6, 8))

    def test_invalid_align_corners(self):
        with helper.raises(tp.TripyException, match="align_corners can only be set with `cubic` or `linear` mode."):
            a = tp.ones((1, 1, 3, 4))
            out = tp.resize(a, "nearest", output_shape=(1, 1, 6, 8), align_corners=True)
