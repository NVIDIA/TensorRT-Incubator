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

import pytest
import tripy as tp

from tripy.flat_ir.ops import DynamicGatherOp
from tripy.flat_ir.function import FlatIRFunction
from tripy.frontend.trace import Trace
import re


class TestGatherOp:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_gather_str(self, axis):
        data = tp.iota((3, 3, 2))
        data.name = "data"
        index = tp.Tensor([1], dtype=tp.int32, name="indices")
        out = tp.gather(data, axis, index)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        func_gather = flat_ir.ops[-1]
        assert isinstance(func_gather, FlatIRFunction)

        gather = func_gather.ops[-1]
        reshape = func_gather.ops[-7]
        print(str(reshape))
        assert isinstance(gather, DynamicGatherOp)
        assert re.match(
            rf"t_inter[0-9]+: \[rank=\(3\), dtype=\(float32\), loc=\(gpu:0\)\] = DynamicGatherOp\(t_inter[0-9]+, t_inter[0-9]+, t_inter[0-9]+, axis={axis}\)",
            str(gather),
        )

    @pytest.mark.parametrize("axis", [0, 1])
    def test_gather_mlir(self, axis):
        out = tp.gather(tp.Tensor([[1, 1, 1], [1, 1, 1]]), axis, tp.Tensor([0], dtype=tp.int32))
        trace = Trace([out])
        flat_ir = trace.to_flat_ir()
        mlir_text = str(flat_ir.to_mlir())
        if axis == 0:
            target = '"stablehlo.dynamic_gather"(%arg0, %arg1, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>}> : (tensor<?x?xi32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xi32>'
        else:
            target = '"stablehlo.dynamic_gather"(%arg0, %arg1, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [0], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>}> : (tensor<?x?xi32>, tensor<?xi32>, tensor<2xi32>) -> tensor<?x?xi32>'
        assert target in mlir_text, mlir_text
