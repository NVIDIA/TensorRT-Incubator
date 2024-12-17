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
import re

import cupy as cp
import numpy as np
import torch

import nvtripy as tp
from tests.helper import raises


class TestStride:
    def test_non_canonical_stride(self):
        test_cases = [
            (
                torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1),
                lambda x: x.contiguous(),
                lambda x: x.clone(memory_format=torch.contiguous_format),
            ),
            (
                cp.arange(12, dtype=cp.float32).reshape(3, 4).transpose(1, 0),
                cp.ascontiguousarray,
                lambda x: x.copy(order="C"),
            ),
            (
                np.arange(12, dtype=np.float32).reshape(3, 4).transpose(1, 0),
                np.ascontiguousarray,
                lambda x: x.copy(order="C"),
            ),
        ]

        for array, contiguous_func, copy_func in test_cases:
            # Test for exception with non-canonical strides
            with pytest.raises(tp.TripyException, match="Non-canonical strides are not supported for Tripy tensors"):
                tp.Tensor(array)

            # Test successful creation with contiguous array
            assert tp.Tensor(contiguous_func(array)) is not None
            assert tp.Tensor(copy_func(array)) is not None
