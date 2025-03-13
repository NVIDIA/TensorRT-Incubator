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
import nvtripy as tp
import pytest
from nvtripy.backend.mlir import memref
from nvtripy.trace.ops.constant import Constant, convert_list_to_array, get_element_type
from tests import helper


def test_get_element_type():
    assert get_element_type([1, 2, 3]) == tp.int32
    assert get_element_type([1.0, 2.0, 3.0]) == tp.float32
    assert get_element_type([[1], [2], [3]]) == tp.int32

    with helper.raises(tp.TripyException, match="Unsupported element type."):
        get_element_type(["a", "b", "c"])


@pytest.mark.parametrize(
    "values, dtype, expected",
    [
        ([True, False, True], tp.bool, b"\x01\x00\x01"),
        ([100000, 200000], tp.int32, b"\xa0\x86\x01\x00@\x0d\x03\x00"),
        ([1, 2], tp.int64, b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"),
        ([1.0, 2.0], tp.float32, b"\x00\x00\x80?\x00\x00\x00@"),
    ],
)
def test_convert_list_to_array(values, dtype, expected):
    result = convert_list_to_array(values, dtype)
    assert result.tobytes() == expected


class TestConstant:
    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_from_small_memref(self, device):
        module = np if device == "cpu" else cp
        data = memref.create_memref_view(module.ones((2, 2), dtype=module.float32))
        constant = Constant(data)
        assert constant.dtype == tp.float32
        assert constant.shape == (2, 2)
        assert constant.device.kind == device

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_from_dlpack_int(self, dtype):
        cp_dtype = cp.int64 if dtype == "int64" else cp.int32
        tripy_dtype = tp.int64 if dtype == "int64" else tp.int32

        data = cp.ones((2, 2), dtype=cp_dtype)
        constant = Constant(data)
        assert constant.dtype == tripy_dtype
        assert constant.shape == (2, 2)
        assert constant.device.kind == "gpu"

    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    def test_from_dlpack_float(self, dtype):
        cp_dtype = cp.float16 if dtype == "float16" else cp.float32
        tripy_dtype = tp.float16 if dtype == "float16" else tp.float32

        data = cp.ones((2, 2), dtype=cp_dtype)
        constant = Constant(data)
        assert constant.dtype == tripy_dtype
        assert constant.shape == (2, 2)
        assert constant.device.kind == "gpu"

    def test_from_list_int(self):
        data = [[1, 2], [3, 4]]
        constant = Constant(data)
        assert constant.dtype == tp.int32
        assert constant.shape == (2, 2)
        assert constant.device.kind == "gpu"

    def test_from_list_float(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        constant = Constant(data)
        assert constant.dtype == tp.float32
        assert constant.shape == (2, 2)
        assert constant.device.kind == "gpu"

    def test_empty_list(self):
        data = [[]]
        constant = Constant(data)
        assert constant.dtype == tp.float32
        assert constant.shape == (1, 0)
        assert constant.device.kind == "gpu"

    def test_infer_rank(self):
        arr = [1.0, 2.0, 3.0]
        t = tp.Tensor(arr)
        assert t.trace_tensor.rank == 1
