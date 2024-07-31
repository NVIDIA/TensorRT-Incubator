#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import tripy as tp


class TestPlugin:
    def test_gelu(self):
        # TODO: We add `+ 1` as a hack to work around MLIR-TRT Issue #915. We should be able to remove it once fixed
        inp = tp.iota((2, 2)) + 1
        out = tp.plugin(
            "CustomGeluPluginDynamic",
            [inp],
            output_info=[(inp.rank, inp.dtype)],
            # Plugin Parameters:
            type_id=0,
        )

        ref_out = tp.gelu(inp)

        assert cp.allclose(cp.from_dlpack(out), cp.from_dlpack(ref_out))

    def test_dynamic_shape_gelu(self):
        def gelu(X):
            return tp.plugin("CustomGeluPluginDynamic", [X], output_info=[(X.rank, X.dtype)], type_id=0)

        compiler = tp.Compiler(gelu)
        compiled_gelu = compiler.compile(tp.InputInfo((2, (1, 2, 3), 4), dtype=tp.float32))

        inp = tp.iota((2, 1, 4))
        assert tp.allclose(compiled_gelu(inp), tp.gelu(inp))

        new_inp = tp.ones((2, 2, 4), dtype=tp.float32)
        assert tp.allclose(compiled_gelu(new_inp), tp.gelu(new_inp))
