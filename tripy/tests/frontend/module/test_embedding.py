#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvtripy as tp
from tests import helper


class TestEmbedding:
    def test_embedding(self):
        embedding = tp.Embedding(20, 30)
        embedding.weight = tp.ones(embedding.weight.shape)

        assert isinstance(embedding, tp.Embedding)
        assert cp.from_dlpack(embedding.weight).get().shape == (20, 30)

    def test_incorrect_input_dtype(self):
        a = tp.ones((2, 3))
        embd = tp.Embedding(4, 16)
        embd.weight = tp.ones(embd.weight.shape)

        with helper.raises(tp.TripyException, match="Invalid inputs for function: 'gather'."):
            out = embd(a)
