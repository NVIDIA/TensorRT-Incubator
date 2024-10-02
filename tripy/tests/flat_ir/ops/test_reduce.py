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
from tripy.frontend.trace import Trace
from tripy.flat_ir.function import FlatIRFunction
from tripy.flat_ir.ops import ArgMinMaxOp, ConvertOp, DivideOp, DynamicBroadcastOp, MulOp, ReduceOp
import re


class TestReduceOp:
    def test_sum_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = tp.sum(inp, 0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_reduce = flat_ir.ops[-1]
        reduce = func_reduce.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(int32\), loc=\(gpu:0\)\] = ReduceOp\(t_inter[0-9]+, t_inter[0-9]+, reduce_mode='sum', reduce_dims=\[0\]\)",
            str(reduce),
        )

    def test_max_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = tp.max(inp, 0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_reduce = flat_ir.ops[-1]
        reduce = func_reduce.ops[-1]
        assert isinstance(reduce, ReduceOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(int32\), loc=\(gpu:0\)\] = ReduceOp\(t_inter[0-9]+, t_inter[0-9]+, reduce_mode='max', reduce_dims=\[0\]\)",
            str(reduce),
        )

    def test_mean_str(self):
        inp = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32, name="inp")
        out = tp.mean(inp, 0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_div = flat_ir.ops[-1]
        div = func_div.ops[-1]
        broadcast_a = func_div.ops[-3]
        broadcast_b = func_div.ops[-2]
        assert isinstance(div, DivideOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = DivideOp\(t_inter[0-9]+, t_inter[0-9]+\)",
            str(div),
        )

        assert isinstance(broadcast_a, DynamicBroadcastOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(t_inter[0-9]+, t_inter[0-9]+, broadcast_dim=\[[0-9]*\]\)",
            str(broadcast_a),
        )

        assert isinstance(broadcast_b, DynamicBroadcastOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicBroadcastOp\(t_inter[0-9]+, t_inter[0-9]+, broadcast_dim=\[[0-9]*\]\)",
            str(broadcast_b),
        )

        mul = flat_ir.ops[-3].ops[-1]
        assert isinstance(mul, MulOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(0\), dtype=\(int32\), loc=\(gpu:0\)\] = MulOp\(t_inter[0-9]+, t_inter[0-9]+\)",
            str(mul),
        )

        func_reduce = flat_ir.ops[1]
        assert isinstance(func_reduce, FlatIRFunction)

        reduce = func_reduce.ops[-1]
        assert isinstance(reduce, ReduceOp)

        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(float32\), loc=\(gpu:0\)\] = ReduceOp\(t_inter[0-9]+, t_inter[0-9]+, reduce_mode='sum', reduce_dims=\[0\]\)",
            str(reduce),
        )

    def test_argmax_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = tp.argmax(inp, 0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_argminmax = flat_ir.ops[-1]
        assert isinstance(func_argminmax, FlatIRFunction)

        argminmax = func_argminmax.ops[-1]
        assert isinstance(argminmax, ArgMinMaxOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(int32\), loc=\(gpu:0\)\] = ArgMinMaxOp\(t_inter[0-9]+, t_inter[0-9]+, t_inter[0-9]+, t_inter[0-9]+, reduce_mode='argmax', reduce_dims=\[0\]\)",
            str(argminmax),
        )

    def test_argmin_str(self):
        inp = tp.Tensor([[1, 2], [3, 4]], name="inp")
        out = tp.argmin(inp, 0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_argminmax = flat_ir.ops[-1]
        assert isinstance(func_argminmax, FlatIRFunction)

        argminmax = func_argminmax.ops[-1]
        assert isinstance(argminmax, ArgMinMaxOp)
        assert re.match(
            r"t_inter[0-9]+: \[rank=\(1\), dtype=\(int32\), loc=\(gpu:0\)\] = ArgMinMaxOp\(t_inter[0-9]+, t_inter[0-9]+, t_inter[0-9]+, t_inter[0-9]+, reduce_mode='argmin', reduce_dims=\[0\]\)",
            str(argminmax),
        )
