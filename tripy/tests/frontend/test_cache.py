# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from tests import helper

import nvtripy as tp
import cupy as cp

from nvtripy.constants import STROGE_OP_CACHE_VOLUME_THRESHOLD
from nvtripy.frontend.trace import Trace
from nvtripy.frontend.cache import ExecutableCache


@pytest.fixture
def cache():
    return ExecutableCache()


@pytest.fixture
def mock_global_cache(monkeypatch, cache):
    monkeypatch.setattr(tp.frontend.cache, "global_cache", cache)
    return cache


class TestCache:
    def test_identical_graph_different_input_shapes(self, mock_global_cache):
        input1 = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float32)
        input2 = tp.Tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=tp.float32)

        layer = tp.Linear(2, 3)

        output1 = layer(input1)
        assert mock_global_cache.get(Trace([output1.trace_tensor]), devices=[output1.device]) is None
        output1.eval()
        assert mock_global_cache.get(Trace([layer(input1).trace_tensor]), devices=[output1.device]) is not None

        output2 = layer(input2)
        assert mock_global_cache.get(Trace([output2.trace_tensor]), devices=[output2.device]) is None
        output2.eval()
        assert mock_global_cache.get(Trace([layer(input2).trace_tensor]), devices=[output2.device]) is not None

    def test_identical_graph_different_input_names(self, mock_global_cache):
        input1 = tp.Tensor([[1.0, 2.0]], dtype=tp.float32, name="input_a")
        input2 = tp.Tensor([[1.0, 2.0]], dtype=tp.float32, name="input_b")

        layer = tp.Linear(2, 3)
        output1 = layer(input1)
        output1.eval()

        output2 = layer(input2)
        assert mock_global_cache.get(Trace([output2.trace_tensor]), devices=[output2.device]) is not None

    def test_identical_graph_different_output_names(self, mock_global_cache):
        input_tensor = tp.Tensor([[1.0, 2.0]], dtype=tp.float32)

        layer = tp.Linear(2, 3)

        output1 = layer(input_tensor)
        output1.name = "output_a"
        output1.eval()

        output2 = layer(input_tensor)
        output2.name = "output_b"
        assert mock_global_cache.get(Trace([output2.trace_tensor]), devices=[output2.device]) is not None

    def test_different_graphs_different_cache_entries(self, mock_global_cache):
        input_tensor = tp.Tensor([[1.0, 2.0]], dtype=tp.float32)

        layer1 = tp.Linear(2, 3)
        layer2 = tp.Linear(2, 4)

        output1 = layer1(input_tensor)
        assert mock_global_cache.get(Trace([output1.trace_tensor]), devices=[output1.device]) is None
        output1.eval()
        assert mock_global_cache.get(Trace([layer1(input_tensor).trace_tensor]), devices=[output1.device]) is not None

        output2 = layer2(input_tensor)
        assert mock_global_cache.get(Trace([output2.trace_tensor]), devices=[output2.device]) is None
        output2.eval()
        assert mock_global_cache.get(Trace([layer2(input_tensor).trace_tensor]), devices=[output2.device]) is not None

    # test change value of small shape tensor to see should fail since it is not lifted
    # test change value of big shape tensor to see should not fail since it is lifted as input
    # test cache_not being used value match

    def test_cache_not_being_used(self, mock_global_cache):
        with helper.config("tripy_eager_cache", False):
            input_tensor = tp.Tensor([[1.0, 2.0]], dtype=tp.float32)

            input_tensor.eval()
            assert mock_global_cache.get(Trace([input_tensor.trace_tensor]), devices=[input_tensor.device]) is None
