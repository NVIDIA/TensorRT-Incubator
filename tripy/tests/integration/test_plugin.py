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

import cupy as cp

import nvtripy as tp


class TestPlugin:
    def test_gelu(self):
        inp = tp.iota((2, 2))
        out = tp.plugin(
            name="CustomGeluPluginDynamic",
            inputs=[inp],
            output_info=[(inp.rank, inp.dtype)],
            # Plugin Parameters:
            type_id=0,
        )

        ref_out = tp.gelu(inp)

        assert out.shape == ref_out.shape == inp.shape
        assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(ref_out), atol=0.001)

    def test_dynamic_shape_gelu(self):
        def gelu(X):
            return tp.plugin(name="CustomGeluPluginDynamic", inputs=[X], output_info=[(X.rank, X.dtype)], type_id=0)

        compiled_gelu = tp.compile(gelu, args=[tp.InputInfo((2, (1, 2, 3), 4), dtype=tp.float32)])

        inp = tp.iota((2, 1, 4))
        out = compiled_gelu(inp)
        assert out.shape == inp.shape
        assert tp.allclose(out, tp.gelu(inp), atol=0.001)

        new_inp = tp.ones((2, 2, 4), dtype=tp.float32)
        out = compiled_gelu(new_inp)
        assert out.shape == new_inp.shape
        assert tp.allclose(out, tp.gelu(new_inp), atol=0.001)
