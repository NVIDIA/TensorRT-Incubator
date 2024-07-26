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
from tripy import int32

from tripy.flat_ir.ops import DynamicGatherOp
from tripy.frontend.trace import Trace


class TestGatherOp:
    def test_gather_str(self):
        data = tp.Tensor([3.0, 4.0], name="data")
        index = tp.Tensor([0], dtype=int32, name="indices")
        out = tp.gather(data, 0, index)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        gather = flat_ir.ops[-1]
        reshape = flat_ir.ops[-2]

        print(str(reshape))
        assert isinstance(gather, DynamicGatherOp)
        assert (
            str(gather)
            == "out: [rank=(1), dtype=(float32), loc=(gpu:0)] = DynamicGatherOp(data, indices, t_inter3, axis=0)"
        )
