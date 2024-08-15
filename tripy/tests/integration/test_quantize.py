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

import cupy as cp
import numpy as np
import pytest
import re
import torch

import tripy as tp
from tests.helper import raises, TORCH_DTYPES
from tests.conftest import skip_if_older_than_sm80, skip_if_older_than_sm89


class TestQuantize:
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_tensor(self, dtype):
        input = torch.tensor([1.0, 2.0], dtype=TORCH_DTYPES[dtype])
        scale = torch.tensor(0.5, dtype=TORCH_DTYPES[dtype])
        input_tp = tp.Tensor(input, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input_tp, scale_tp, tp.int8)
        expected = (input / scale).to(dtype=torch.int8)
        assert torch.equal(expected, torch.from_dlpack(quantized).to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_channel(self, dtype):
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=TORCH_DTYPES[dtype])
        scale = torch.tensor([0.2, 0.1], dtype=TORCH_DTYPES[dtype])
        input_tp = tp.Tensor(input, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input_tp, scale_tp, tp.int8, dim=0)
        expected = (input / scale.reshape(2, 1)).to(dtype=torch.int8)
        assert torch.equal(expected, torch.from_dlpack(quantized).to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_tensor(self, dtype):
        input = torch.tensor([1.0, 2.0], dtype=TORCH_DTYPES[dtype])
        scale = torch.tensor(0.5, dtype=TORCH_DTYPES[dtype])
        input_tp = tp.Tensor(input, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input_tp, scale_tp, tp.float8)
        assert quantized.dtype == tp.float8
        expected = (input / scale).to(dtype=torch.float32)
        with raises(
            Exception,
            match=re.escape("UNIMPLEMENTED: Invalid or unsupported DLPack float width: 8 bits"),
        ):
            assert torch.equal(expected, torch.from_dlpack(quantized).to(dtype=torch.float32).to("cpu"))
        assert torch.equal(expected, torch.from_dlpack(tp.cast(quantized, dtype=tp.float32)).to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_channel(self, dtype):
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=TORCH_DTYPES[dtype])
        scale = torch.tensor([0.2, 0.1], dtype=TORCH_DTYPES[dtype])
        input_tp = tp.Tensor(input, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input_tp, scale_tp, tp.float8, dim=0)
        assert quantized.dtype == tp.float8
        expected = (input / scale.reshape(2, 1)).to(dtype=torch.float32)
        with raises(
            Exception,
            match=re.escape("UNIMPLEMENTED: Invalid or unsupported DLPack float width: 8 bits"),
        ):
            assert torch.equal(expected, torch.from_dlpack(quantized).to(dtype=torch.float32).to("cpu"))
        assert torch.equal(expected, torch.from_dlpack(tp.cast(quantized, dtype=tp.float32)).to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @pytest.mark.parametrize("quant_mode", ["block-wise", "per-tensor", "per-channel-0", "per-channel-1"])
    def test_qdq_int4(self, dtype, quant_mode):
        if quant_mode == "block-wise":
            dim = None
            scale = torch.ones((2, 4), dtype=TORCH_DTYPES[dtype])
        elif quant_mode == "per-tensor":
            dim = None
            scale = torch.tensor(1.0, dtype=TORCH_DTYPES[dtype])
        elif quant_mode.endswith("0"):
            dim = 0
            scale = torch.ones((4,), dtype=TORCH_DTYPES[dtype])
        elif quant_mode.endswith("1"):
            dim = 1
            scale = torch.ones((4,), dtype=TORCH_DTYPES[dtype])

        input = torch.ones((4, 4), dtype=TORCH_DTYPES[dtype])
        input_tp = tp.Tensor(input, dtype=dtype)
        scale_tp = tp.Tensor(scale)
        quantized = tp.quantize(input_tp, scale_tp, tp.int4, dim)
        dequantized = tp.dequantize(quantized, scale_tp, dtype, dim)
        assert torch.equal(input, torch.from_dlpack(dequantized).to("cpu"))

    def test_negative_non_constant_scale(self):
        input = tp.ones((4, 4))
        scale = tp.ones((4,))
        quantized = tp.quantize(input, scale, tp.int8, dim=0)
        with raises(
            tp.TripyException,
            match="Scale must be a constant tensor in quantize op",
        ):
            print(quantized)
