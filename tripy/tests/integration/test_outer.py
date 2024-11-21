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
import pytest

import tripy as tp
import torch


class TestOuter:
    def test_outer(self, eager_or_compiled):
        v1 = tp.arange(5, dtype=tp.float32)
        v2 = tp.arange(4, dtype=tp.float32)
        output = eager_or_compiled(tp.outer, v1, v2)

        t1 = torch.arange(5, dtype=torch.float32)
        t2 = torch.arange(4, dtype=torch.float32)
        torch_out = torch.outer(t1, t2)
        assert output.shape == list(torch_out.shape)
        assert tp.allclose(output, tp.Tensor(torch_out))

    def test_empty(self, eager_or_compiled):
        v1 = tp.Tensor([])
        v2 = tp.arange(3, dtype=tp.float32)
        output = eager_or_compiled(tp.outer, v1, v2)

        assert output.shape == [0, 3]
