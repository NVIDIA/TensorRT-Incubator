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


class TestFlatIR:
    def test_tensor_connectivity(self):
        # When we build up a FlatIR with multiple layers, the tensors/ops
        # should be connected to each other - i.e. the producer/inputs fields
        # should let you walk through the entire FlatIR.
        inp = tp.Tensor([0])

        b = tp.tanh(inp)
        out = tp.tanh(b)

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        # Check that `b` is connected to `inp`
        assert flat_ir.ops[1].inputs[0].producer is flat_ir.ops[0]
        assert flat_ir.ops[1].inputs[0] is flat_ir.ops[0].outputs[0]

        # Check that `out` is connected to `b`
        assert flat_ir.ops[2].inputs[0].producer is flat_ir.ops[1]
        assert flat_ir.ops[2].inputs[0] is flat_ir.ops[1].outputs[0]
