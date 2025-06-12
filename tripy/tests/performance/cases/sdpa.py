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
import math

import nvtripy as tp
import torch
from tests.performance.conftest import PerfParam, perf_fixture


@perf_fixture(
    params=[
        PerfParam(tp.float32, perf_threshold=1.25),
        PerfParam(tp.float16, perf_threshold=1.35),
    ]
)
def sdpa(tripy_dtype, torch_dtype):
    class SDPA(tp.Module):
        def forward(self, query, key, value):
            embedding_dim = query.shape[-1]
            qk = query @ (tp.transpose(key, -2, -1) / tp.sqrt(tp.cast(embedding_dim, query.dtype)))
            return tp.cast(tp.softmax(qk, -1), query.dtype) @ value

    class TorchSDPA(torch.nn.Module):
        def forward(self, query, key, value):
            embedding_dim = query.shape[-1]
            qk = query @ (key.transpose(-2, -1) / math.sqrt(embedding_dim))
            return torch.softmax(qk, dim=-1) @ value

    tripy_block = SDPA()
    torch_block = TorchSDPA()

    batch = tp.NamedDimension("batch", 1, 2, 2)
    input_infos = {
        "query": tp.InputInfo(shape=(batch, 1, 4096, 256), dtype=tripy_dtype),
        "key": tp.InputInfo(shape=(batch, 1, 4096, 256), dtype=tripy_dtype),
        "value": tp.InputInfo(shape=(batch, 1, 4096, 256), dtype=tripy_dtype),
    }

    return tripy_block, torch_block, input_infos
