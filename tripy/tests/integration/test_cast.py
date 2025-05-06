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

import cupy as cp
import numpy as np
import pytest

import nvtripy as tp
from tests.conftest import skip_if_older_than_sm89
from tests.helper import NUMPY_TO_TRIPY


class TestCast:
    @pytest.mark.parametrize(
        "input_dtype, target_dtype",
        [
            (np.int32, np.float32),
            (np.float32, np.int32),
            (np.int32, np.int8),
            (np.float32, np.int8),
            (np.int8, np.int32),
            (np.int8, np.float32),
            # important to test conversion into bool because default StableHLO semantics
            # are simply to truncate to i1, which is not desirable
            (np.float32, bool),
            (np.int32, bool),
            # requires a dequantization first
            # TODO(#219): Dequantize fails with dynamic shapes
            # (np.int8, bool),
        ],
    )
    def test_cast(self, input_dtype, target_dtype, eager_or_compiled):
        tp_input_dtype = NUMPY_TO_TRIPY[input_dtype]
        tp_target_dtype = NUMPY_TO_TRIPY[target_dtype]

        # TODO(#222): Integer casts with negative numbers fail in many cases
        input_tensor = tp.Tensor([0, 1, 2], dtype=tp_input_dtype).eval()

        output = eager_or_compiled(tp.cast, input_tensor, tp_target_dtype)

        np_input = np.from_dlpack(tp.copy(input_tensor, device=tp.device("cpu")))
        assert np.array_equal(np.from_dlpack(tp.copy(output, device=tp.device("cpu"))), np_input.astype(target_dtype))

    # these dtypes don't have analogues in numpy
    @pytest.mark.parametrize("source_dtype", [pytest.param(tp.float8, marks=skip_if_older_than_sm89), tp.int4])
    def test_cast_quantized_dtypes_into_bool(self, source_dtype, eager_or_compiled):
        # TODO(#223): Using an odd size leads to a strange crash, so can't just use [-1.0, 0.0, 1.0]
        input_tensor = tp.Tensor([-1.0, 0.0, 0.0, 1.0], dtype=tp.float32)

        def func(input):
            q = tp.quantize(input, scale=1.0, dtype=source_dtype)
            output = tp.cast(q, tp.bool)
            return output

        output = eager_or_compiled(func, input_tensor)
        assert np.from_dlpack(tp.copy(output, device=tp.device("cpu"))).tolist() == [True, False, False, True]

    @pytest.mark.parametrize("target_dtype", [np.float32, np.int32, np.int8])
    def test_cast_from_bool(self, target_dtype, eager_or_compiled):
        tp_target_dtype = NUMPY_TO_TRIPY[target_dtype]

        # In principle, it is not important what *specific* values we convert to,
        # so long as false is mapped to 0 and true to nonzero
        input_tensor = tp.Tensor([False, True])
        np_input = np.from_dlpack(input_tensor)

        output = eager_or_compiled(tp.cast, input_tensor, tp_target_dtype)

        tp_compare_to_zero = np.from_dlpack(tp.copy(output, device=tp.device("cpu"))) == 0

        np_compare_to_zero = np_input.astype(target_dtype) == 0
        assert np.array_equal(tp_compare_to_zero, np_compare_to_zero)
