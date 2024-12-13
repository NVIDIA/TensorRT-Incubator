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
from tests import helper

import tripy as tp


class TestLinear:
    def test_linear_params(self):
        linear = tp.Linear(20, 30)
        assert isinstance(linear, tp.Linear)
        assert linear.weight.shape == [30, 20]
        assert linear.bias.shape == [30]

    def test_mismatched_input_shapes(self):
        a = tp.ones((2, 3))
        linear = tp.Linear(2, 128)
        out = linear(a)

        with helper.raises(
            tp.TripyException, match="contracting dimension sizes must match for lhs/rhs", has_stack_info_for=[a]
        ):
            out.eval()

    @pytest.mark.parametrize("quant_dtype", [tp.int8, tp.float8])
    @pytest.mark.parametrize("weight_quant_dim", [None, 0, 1])
    def test_quantized_params(self, quant_dtype, weight_quant_dim):
        qlinear = tp.Linear(
            20,
            30,
            quant_dtype=quant_dtype,
            weight_quant_dim=weight_quant_dim,
        )
        assert isinstance(qlinear, tp.Linear)
        assert qlinear.dtype == tp.float32
        assert qlinear.quant_dtype == quant_dtype
        assert qlinear.weight.shape == [30, 20]
        assert qlinear.bias.shape == [30]
        assert qlinear.weight_quant_dim == weight_quant_dim
        assert isinstance(qlinear.weight_scale, tp.Tensor)
        assert isinstance(qlinear.input_scale, tp.Tensor)

    def test_load_quantized_params_from_state_dict(self):
        qlinear = tp.Linear(
            20,
            30,
            quant_dtype=tp.int8,
            weight_quant_dim=0,
        )

        qlinear.load_state_dict(
            {"weight_scale": tp.ones((30,)), "input_scale": tp.ones((20,))},
            strict=False,
        )
