
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

import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConcatenateOp


class TestConcatOp:
    def test_str(self):
        a = tp.ones((2, 3))
        a.name = "a"
        b = tp.ones((3, 3))
        b.name = "b"
        out = tp.concatenate([a, b], dim=0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        concat = flat_ir.ops[-1]
        assert isinstance(concat, ConcatenateOp)
        assert str(concat) == "out: [rank=(2), dtype=(float32), loc=(gpu:0)] = ConcatenateOp(a, b, dim=0)"
