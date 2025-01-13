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
import cupy as cu

from nvtripy.constants import STORAGE_OP_CACHE_VOLUME_THRESHOLD
from nvtripy.frontend.trace import Trace
from nvtripy.frontend.cache import ExecutableCache


@pytest.fixture
def mock_global_cache(monkeypatch):
    cache = ExecutableCache()
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

    def test_different_small_input_values_not_lifted(self, mock_global_cache):
        shape = (1, 2)
        input1 = tp.ones(shape, dtype=tp.float32)
        input2 = tp.zeros(shape, dtype=tp.float32)

        layer = tp.Linear(2, 3)

        output1 = layer(input1)
        output1.eval()

        output2 = layer(input2)
        assert mock_global_cache.get(Trace([output2.trace_tensor]), devices=[output2.device]) is None
        output2.eval()
        assert mock_global_cache.get(Trace([layer(input2).trace_tensor]), devices=[output2.device]) is not None

    def test_different_large_input_values_lifted(self, mock_global_cache):
        shape = (STORAGE_OP_CACHE_VOLUME_THRESHOLD, 2)
        input1 = tp.Tensor(cu.ones(shape, dtype=cu.float32), dtype=tp.float32)
        input2 = tp.Tensor(cu.zeros(shape, dtype=cu.float32), dtype=tp.float32)

        layer = tp.Linear(2, 3)

        output1 = layer(input1)
        output1.eval()

        output2 = layer(input2)
        output2_trace = Trace([output2.trace_tensor], inputs=Trace._collect_storage_tensors(output2.trace_tensor))
        assert mock_global_cache.get(output2_trace, devices=[output2.device]) is not None

    def test_different_graphs(self, mock_global_cache):
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

    def test_cache_not_being_used_empty_cache(self, mock_global_cache):
        with helper.config("use_cache_in_eager_mode", False):
            tensor1 = tp.ones((1, 1), dtype=tp.float32)

            tensor1.eval()
            assert mock_global_cache.get(Trace([tensor1.trace_tensor]), devices=[tensor1.device]) is None

    def test_cache_not_being_used_value_match(self, mock_global_cache):
        tensor1 = tp.ones((1, 1), dtype=tp.float32)
        tensor1.eval()

        tensor1_cached = tp.ones((1, 1), dtype=tp.float32)
        assert mock_global_cache.get(Trace([tensor1_cached.trace_tensor]), devices=[tensor1.device]) is not None
        tensor1_cached.eval()

        with helper.config("use_cache_in_eager_mode", False):
            tensor2 = tp.ones((1, 1), dtype=tp.float32)
            tensor2.eval()

        assert tp.equal(tensor1_cached, tensor2)
