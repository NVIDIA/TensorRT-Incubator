
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

import pytest
import re

import tripy as tp
from tests import helper
from tripy.flat_ir.ops import DynamicIotaOp
from tripy.frontend.trace import Trace


@pytest.fixture
def flat_ir():
    out = tp.iota((2, 3))
    out.name = "out"

    trace = Trace([out])
    yield trace.to_flat_ir()


class TestIotaOp:
    def test_str(self, flat_ir):
        iota = flat_ir.ops[-1]
        assert isinstance(iota, DynamicIotaOp)
        assert re.match(
            r"out: \[rank=\(2\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicIotaOp\(t[0-9]+, dim=0\)",
            str(iota),
        )

    def test_mlir(self, flat_ir):
        print(str(flat_ir.to_mlir()))
        helper.check_mlir(
            flat_ir.to_mlir(),
            """
            module {
                func.func @main() -> tensor<?x?xf32> {
                    %c = stablehlo.constant dense<[2, 3]> : tensor<2xi32>
                    %0 = stablehlo.dynamic_iota %c, dim = 0 : (tensor<2xi32>) -> tensor<?x?xf32>
                    return %0 : tensor<?x?xf32>
                }
            }
            """,
        )
