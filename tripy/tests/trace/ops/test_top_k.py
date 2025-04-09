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
import nvtripy as tp
import pytest
from nvtripy.trace.ops.top_k import TopKMax, TopKMin


@pytest.mark.parametrize("TopKType", [TopKMax, TopKMin])
class TestTopK:
    def test_infer_rank(self, TopKType):
        inp = tp.ones((2, 2, 3))
        values, indices = TopKType([inp.trace_tensor], dim=2, k=2).outputs
        assert values.rank == inp.rank
        assert indices.rank == inp.rank

    def test_infer_dtypes(self, TopKType):
        inp = tp.ones((2, 2, 3))
        values, indices = TopKType([inp.trace_tensor], dim=2, k=2).outputs
        assert values.dtype == inp.dtype
        assert indices.dtype == tp.int32

    def test_infer_devices(self, TopKType):
        inp = tp.ones((2, 2, 3))
        values, indices = TopKType([inp.trace_tensor], dim=2, k=2).outputs
        assert values.device == inp.device
        assert indices.device == inp.device
