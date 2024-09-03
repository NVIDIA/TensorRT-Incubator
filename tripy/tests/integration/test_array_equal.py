#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
import torch

import tripy as tp


class TestArrayEqual:
    @pytest.mark.parametrize(
        "a, b",
        [
            (tp.Tensor([1, 2], dtype=tp.float32), tp.Tensor([1, 2], dtype=tp.float32)),
            (tp.ones((2, 2), dtype=tp.int32), tp.Tensor([[1, 1], [1, 1]], dtype=tp.int32)),
            (tp.ones((1, 4)), tp.ones((4, 1))),
        ],
    )
    def test_array_equal(self, a, b):
        torch_result = torch.equal(torch.from_dlpack(a), torch.from_dlpack(b))
        tp_result = tp.array_equal(a, b)
        assert torch_result == tp_result
