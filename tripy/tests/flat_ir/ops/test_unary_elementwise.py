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
import pytest

from tests import helper
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ExpOp, TanhOp, RsqrtOp, LogOp, SineOp, CosineOp, SqrtOp, AbsOp


_UNARY_OPS = {
    tp.exp: ("ExpOp", ExpOp, "stablehlo.exponential"),
    tp.tanh: ("TanhOp", TanhOp, "stablehlo.tanh"),
    tp.rsqrt: ("RsqrtOp", RsqrtOp, "stablehlo.rsqrt"),
    tp.log: ("LogOp", LogOp, "stablehlo.log"),
    tp.sin: ("SineOp", SineOp, "stablehlo.sine"),
    tp.cos: ("CosineOp", CosineOp, "stablehlo.cosine"),
    tp.sqrt: ("SqrtOp", SqrtOp, "stablehlo.sqrt"),
    tp.abs: ("AbsOp", AbsOp, "stablehlo.abs"),
}


@pytest.fixture
def flat_ir(request):
    tp_func = request.param
    out = tp_func(tp.Tensor([1.0, 2.0], name="inp"))
    out.name = "out"

    trace = Trace([out])
    flat_ir = trace.to_flat_ir()
    yield flat_ir


class TestUnaryElementWiseOps:
    @pytest.mark.parametrize(
        "flat_ir, op_detail", [(tp_func, op_detail) for tp_func, op_detail in _UNARY_OPS.items()], indirect=["flat_ir"]
    )
    def test_str(self, flat_ir, op_detail):
        op_type = flat_ir.ops[-1]
        assert isinstance(op_type, op_detail[1])
        assert str(op_type) == f"out: [rank=(1), dtype=(float32), loc=(gpu:0)] = {op_detail[0]}(inp)"

    @pytest.mark.parametrize(
        "flat_ir, op_detail", [(tp_func, op_detail) for tp_func, op_detail in _UNARY_OPS.items()], indirect=["flat_ir"]
    )
    def test_mlir(self, flat_ir, op_detail):
        template = """
            module {{
                func.func @main() -> tensor<2xf32> {{
                    %cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
                    %0 = {op_type} %cst : tensor<2xf32>
                    return %0 : tensor<2xf32>
                }}
            }}
            """

        helper.check_mlir(flat_ir.to_mlir(), template.format(op_type=op_detail[2]))
