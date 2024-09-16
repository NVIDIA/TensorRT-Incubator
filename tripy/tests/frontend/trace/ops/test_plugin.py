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
from tripy.frontend.trace.ops import Plugin


class TestPlugin:
    def test_op(self):
        X = tp.iota((1, 2, 4, 4))
        rois = tp.Tensor([[0.0, 0.0, 9.0, 9.0], [0.0, 5.0, 4.0, 9.0]], dtype=tp.float32)
        batch_indices = tp.zeros((2,), dtype=tp.int32)

        out = tp.plugin(
            "ROIAlign_TRT", [X, rois, batch_indices], output_info=[(X.rank, X.dtype)], output_height=5, output_width=5
        )

        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, Plugin)
