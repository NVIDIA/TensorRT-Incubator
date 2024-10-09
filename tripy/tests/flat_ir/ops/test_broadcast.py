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

import re
import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import DynamicBroadcastOp


class TestBroadcastOp:
    def test_str(self):
        out = tp.full([2, 3], value=1.0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_broadcast = flat_ir.ops[-1]
        broadcast = func_broadcast.ops[-1]
        assert isinstance(broadcast, DynamicBroadcastOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(2\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(t_inter[0-9]+, t_inter[0-9]+, broadcast_dim=\[\]\)",
            str(broadcast),
        )
