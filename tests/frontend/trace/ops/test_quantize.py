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

import numpy as np
import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Quantize


class TestQuantize:
    def test_op(self):
        a = tp.Tensor([1.0, 2.0])
        a = tp.quantize(a, 0.9, tp.int8)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Quantize)

    def test_infer_rank(self):
        a = tp.ones((2, 3))
        a = tp.quantize(a, 0.9, tp.int8)
        assert a.trace_tensor.rank == 2

    def test_invalid_input_dtype(self):
        a = tp.Tensor([1, 2], dtype=tp.int32)
        with helper.raises(
            tp.TripyException,
            match="Input does not have a valid dtype in quantize op.",
        ):
            a = tp.quantize(a, 0.9, tp.int8)

    def test_unsupported_quant_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="Unsupported dtype in quantize op.",
        ):
            a = tp.quantize(a, 0.9, tp.float16)

    def test_invalid_scale_per_channel(self):
        a = tp.ones((4, 4))
        scale = 0.5
        with helper.raises(
            tp.TripyException,
            match="If dim is given, scale must be a 1-D tensor in per-channel quantize op",
        ):
            a = tp.quantize(a, scale, tp.int8, dim=0)

    def test_invalid_input_blockwise(self):
        a = tp.ones((4,))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Input must be a 2-D tensor in block-wise quantize op",
        ):
            a = tp.quantize(a, scale, tp.int4)

    def test_unsupported_blockwise_dtype(self):
        a = tp.ones((4, 4))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Unsupported dtype in block-wise quantize op",
        ):
            a = tp.quantize(a, scale, tp.int8)

    def test_invalid_scale_per_tensor(self):
        a = tp.ones((4, 4))
        scale = [0.5, 0.5]
        with helper.raises(
            tp.TripyException,
            match="Scale must be a scalar tensor in per-tensor quantize op",
        ):
            a = tp.quantize(a, scale, tp.int8)
