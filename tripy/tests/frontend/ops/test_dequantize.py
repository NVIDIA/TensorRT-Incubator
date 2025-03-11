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

import numpy as np

import nvtripy as tp
from tests import helper


class TestDequantize:
    def test_invalid_input_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="Unsupported data type in 'dequantize'",
        ):
            a = tp.dequantize(a, 0.9, tp.float32)

    def test_invalid_dequant_dtype(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        with helper.raises(
            tp.TripyException,
            match="Unsupported data type in 'dequantize'",
        ):
            a = tp.dequantize(a, 1, tp.int32)

    def test_invalid_scale_per_channel(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        scale = 0.5
        with helper.raises(
            tp.TripyException,
            match="If dim is given, scale must be a 1-D tensor in per-channel dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32, dim=0)

    def test_invalid_input_blockwise(self):
        a = tp.Tensor(np.ones((4,), dtype=np.int8))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Input must be a 2-D tensor in block-wise dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)

    def test_unsupported_blockwise_dtype(self):
        a = tp.Tensor(np.ones((4, 4), dtype=np.int8))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Unsupported data type in block-wise dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)

    def test_invalid_scale_per_tensor(self):
        a = tp.Tensor(np.ones((4, 4), dtype=np.int8))
        scale = tp.Tensor([0.5] * 4)
        with helper.raises(
            tp.TripyException,
            match="Scale must be a scalar tensor in per-tensor dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)

    def test_non_constant_scale(self):
        input = tp.ones((4, 4), dtype=tp.int8)
        scale = tp.ones((4,))
        with helper.raises(
            tp.TripyException,
            match="Scale must be a constant"
        ):
            dequantized = tp.dequantize(input, scale, tp.float32, dim=0)
