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

import tripy as tp
from tests.helper import raises


class TestStride:
    def assert_error_message(self, excinfo, tensor_type, expected_suggestion):
        error_message = str(excinfo.value)
        assert "Non-canonical strides are not supported for Tripy tensors." in error_message
        assert f"For {tensor_type}, use {expected_suggestion}" in error_message

    def tripy_byte_order_strides(self, data):
        return tuple(s * data.dtype.itemsize for s in tp.Tensor(data).stride())

    def test_non_canonical_stride(self):
        # PyTorch test
        t_torch = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a_torch = t_torch.transpose(0, 1)
        with pytest.raises(tp.TripyException) as excinfo:
            tp.Tensor(a_torch)
        self.assert_error_message(excinfo, "PyTorch Tensor", "tensor.contiguous() or tensor.clone()")

        # No exception is thrown.
        print(tp.Tensor(a_torch.contiguous()))
        print(tp.Tensor(a_torch.clone(memory_format=torch.contiguous_format)))

        # CuPy test
        t_cupy = cp.arange(12, dtype=cp.float32).reshape(3, 4)
        a_cupy = t_cupy.transpose(1, 0)
        with pytest.raises(tp.TripyException) as excinfo:
            tp.Tensor(a_cupy)
        self.assert_error_message(excinfo, "CuPy Array", "cp.ascontiguousarray(array) or array.copy(order='C')")

        print(tp.Tensor(cp.ascontiguousarray(a_cupy)))
        print(tp.Tensor(a_cupy.copy(order="C")))

        # NumPy test
        t_numpy = np.arange(12, dtype=np.float32).reshape(3, 4)
        a_numpy = t_numpy.transpose(1, 0)
        with pytest.raises(tp.TripyException) as excinfo:
            tp.Tensor(a_numpy)
        self.assert_error_message(excinfo, "NumPy Array", "np.ascontiguousarray(array) or array.copy(order='C')")

        print(tp.Tensor(np.ascontiguousarray(a_numpy)))
        print(tp.Tensor(a_numpy.copy(order="C")))
