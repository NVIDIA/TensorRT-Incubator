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

import pytest
import torch

import nvtripy as tp
from tests.helper import TORCH_DTYPES
from tests.conftest import skip_if_older_than_sm80, skip_if_older_than_sm89


class TestDequantize:
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_dequantize_int8_per_tensor(self, dtype, eager_or_compiled):
        data = [4, 8]
        input_tp = tp.cast(tp.Tensor(data), tp.int8)
        scale = torch.tensor(0.5, dtype=TORCH_DTYPES[dtype])
        scale_tp = tp.Tensor(scale)

        def func(input):
            return tp.dequantize(input, scale_tp, dtype)

        dequantized = eager_or_compiled(func, input_tp)
        expected = torch.tensor(data) * scale
        output = torch.from_dlpack(dequantized)
        assert torch.allclose(expected, output.to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_dequantize_int8_per_channel(self, dtype, eager_or_compiled):
        # TODO: Fix in #153
        if dtype == tp.float16:
            pytest.skip("TRT does not support fp16->int8 per-channel dequant.")
        data = [[4, 8], [4, 8]]
        input_tp = tp.cast(tp.Tensor(data), tp.int8)
        scale = torch.tensor([0.8, 0.9], dtype=TORCH_DTYPES[dtype])
        scale_tp = tp.Tensor(scale)

        def func(input):
            return tp.dequantize(input, scale_tp, dtype, dim=0)

        dequantized = eager_or_compiled(func, input_tp)
        expected = torch.tensor(data) * scale.reshape((2, 1))
        output = torch.from_dlpack(dequantized)
        assert torch.allclose(expected, output.to("cpu"))

    # TODO(#161): Update float8 test to use frontend representation
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_float8_per_tensor(self, dtype):
        data_value = [1.0, 1.0]
        input_tp = tp.cast(tp.Tensor(data_value), dtype=tp.float8)
        scale = torch.tensor(0.5, dtype=TORCH_DTYPES[dtype])
        scale_tp = tp.Tensor(scale)
        dequantized = tp.dequantize(input_tp, scale_tp, dtype)
        assert dequantized.dtype == dtype
        print(dequantized)
        expected = torch.Tensor(data_value) * scale
        output = torch.from_dlpack(dequantized).to(dtype=torch.float32)
        assert torch.allclose(expected, output.to("cpu"))

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_float8_per_channel(self, dtype):
        data_value = [[1.0, 1.0], [1.0, 1.0]]
        input_tp = tp.cast(tp.Tensor(data_value), dtype=tp.float8)
        scale = torch.tensor([0.8, 0.9], dtype=TORCH_DTYPES[dtype])
        scale_tp = tp.Tensor(scale)
        dequantized = tp.dequantize(input_tp, scale_tp, dtype, dim=0)
        assert dequantized.dtype == dtype
        print(dequantized)
        expected = torch.Tensor(data_value) * scale.reshape((2, 1))
        output = torch.from_dlpack(dequantized).to(dtype=torch.float32)
        assert torch.allclose(expected, output.to("cpu"))
