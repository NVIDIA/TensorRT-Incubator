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

import numpy as np
import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConstantOp


class TestConstantOp:
    def test_str(self):
        out = tp.Tensor([2.0, 3.0], dtype=tp.float32, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        const = flat_ir.ops[-1]
        assert isinstance(const, ConstantOp)
        assert str(const) == "out: [rank=(1), shape=((2,)), dtype=(float32), loc=(gpu:0)] = ConstantOp(data=[2.0, 3.0])"

    def test_mlir(self):
        out = tp.Tensor([2, 3], dtype=tp.int32, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()
        mlir_text = str(flat_ir.to_mlir())
        target = "%c = stablehlo.constant dense<[2, 3]> : tensor<2xi32>"
        assert target in mlir_text

    def test_mlir_bool(self):
        # we need to create a bool constant with an int constant and then cast because MLIR does not allow
        # for bools in dense array attrs
        data_np = np.array([True, False])
        out = tp.Tensor(data_np, dtype=tp.bool, name="out")

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()
        mlir_text = str(flat_ir.to_mlir())

        int_constant = "%c = stablehlo.constant dense<[1, 0]> : tensor<2xi32>"
        conversion = "%0 = stablehlo.convert %c : (tensor<2xi32>) -> tensor<2xi1>"
        assert int_constant in mlir_text
        assert conversion in mlir_text
