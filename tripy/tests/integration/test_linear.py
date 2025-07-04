#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import cupy as cp
import numpy as np
import pytest

import nvtripy as tp
from tests import helper
from tests.conftest import skip_if_older_than_sm89


class TestLinear:
    def test_linear_module(self, eager_or_compiled):
        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(4, 2)

            def forward(self, x):
                return self.linear(x)

        net = Network()
        net.linear.weight = tp.iota((2, 4))
        net.linear.bias = tp.iota((2,))
        cp_a1 = cp.ones((3, 4), dtype=cp.float32)
        a1 = tp.Tensor(cp_a1)

        out = eager_or_compiled(net, a1)

        np_weight = cp.from_dlpack(net.linear.weight).get()
        np_bias = cp.from_dlpack(net.linear.bias).get()

        np_out = cp_a1.get() @ (np_weight.transpose()) + np_bias

        assert (cp.from_dlpack(out).get() == np.array(np_out)).all()


class TestQuantLinear:
    def _create_network(self, use_input_scale, quant_dtype, weight_quant_dim):
        out_feat = 2
        in_feat = 4

        def _get_dummy_scale(quant_dim):
            if quant_dim is None:
                scale = 1.0
            elif quant_dim == 0:
                scale = [1.0] * out_feat
            elif quant_dim == 1:
                scale = [1.0] * in_feat
            return tp.Tensor(scale)

        class Network(tp.Module):
            def __init__(self):
                super().__init__()
                self.linear = tp.Linear(
                    in_feat,
                    out_feat,
                    quant_dtype=quant_dtype,
                    weight_quant_dim=weight_quant_dim,
                )

            def forward(self, x):
                return self.linear(x)

        net = Network()
        net.linear.weight = tp.iota((out_feat, in_feat))
        net.linear.bias = tp.iota((out_feat,))

        net.linear.weight_scale = _get_dummy_scale(weight_quant_dim)
        if use_input_scale:
            net.linear.input_scale = _get_dummy_scale(None)
        return net

    @pytest.mark.parametrize("use_input_scale", [False, True])
    @pytest.mark.parametrize("quant_dtype", [tp.int8, pytest.param(tp.float8, marks=skip_if_older_than_sm89)])
    @pytest.mark.parametrize("weight_quant_dim", [None, 0, 1])
    def test_quant_linear(self, use_input_scale, quant_dtype, weight_quant_dim, eager_or_compiled):
        net = self._create_network(use_input_scale, quant_dtype, weight_quant_dim)

        cp_a1 = cp.ones((3, 4), dtype=cp.float32)
        a1 = tp.Tensor(cp_a1)
        if use_input_scale and weight_quant_dim == 1:
            with helper.raises(
                tp.TripyException,
                match="Unsupported quantization parameters for Linear module.",
            ):
                out = eager_or_compiled(net, a1)
        else:
            out = eager_or_compiled(net, a1)

            np_weight = cp.from_dlpack(net.linear.weight).get()
            np_bias = cp.from_dlpack(net.linear.bias).get()

            np_out = cp_a1.get() @ (np_weight.transpose()) + np_bias

            assert (cp.from_dlpack(out).get() == np.array(np_out)).all()

    @pytest.mark.parametrize(
        "weight_quant_dim, scale",
        [
            (None, np.ones((2, 4), dtype=np.float32)),
            (None, 1.0),
            (0, np.ones((8,), dtype=np.float32)),
            (1, np.ones((4,), dtype=np.float32)),
        ],
        ids=["block-wise", "per-tensor", "per-channel-0", "per-channel-1"],
    )
    def test_quant_linear_int4_weight_only(self, weight_quant_dim, scale, eager_or_compiled):
        scale = tp.Tensor(scale)

        # HACK: Use ones for stable accuracy.
        np_weight = np.ones((8, 4), dtype=np.float32)
        np_bias = np.ones((8,), dtype=np.float32)

        linear = tp.Linear(4, 8, quant_dtype=tp.int4, weight_quant_dim=weight_quant_dim)
        linear.weight_scale = scale
        linear.weight = tp.Tensor(np_weight)
        linear.bias = tp.Tensor(np_bias)

        cp_input = cp.ones((4, 4), dtype=np.float32)
        input = tp.Tensor(cp_input)
        out = eager_or_compiled(linear, input)

        np_out = cp_input.get() @ (np_weight.transpose()) + np_bias

        assert np.array_equal(cp.from_dlpack(out).get(), np_out)
