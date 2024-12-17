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
import cupy as cp
import numpy as np

import nvtripy as tp


from nvtripy.backend.mlir import memref
from nvtripy.frontend.trace.ops import Storage
from nvtripy.frontend.trace.tensor import TraceTensor


class TestStorage:

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_from_memref(self, device):
        module = np if device == "cpu" else cp
        data = memref.create_memref_view(module.ones((2, 2), dtype=module.float32))
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (2, 2)
        assert storage.device.kind == device
        assert storage.data_str == ""

    @pytest.mark.parametrize("dtype", ["int64", "int32"])
    def test_from_dlpack_int(self, dtype):
        cp_dtype = cp.int64 if dtype == "int64" else cp.int32
        tripy_dtype = tp.int64 if dtype == "int64" else tp.int32

        data = cp.ones((2, 2), dtype=cp_dtype)
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tripy_dtype
        assert storage.shape == (2, 2)
        assert storage.device.kind == "gpu"
        assert storage.data_str == "[[1 1]\n [1 1]]"

    def test_from_dlpack_float(self):
        data = cp.ones((2, 2), dtype=cp.float32)
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (2, 2)
        assert storage.device.kind == "gpu"
        assert storage.data_str == "[[1. 1.]\n [1. 1.]]"

    def test_from_list_int(self):
        data = [[1, 2], [3, 4]]
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.int32
        assert storage.shape == (2, 2)
        assert storage.device.kind == "gpu"
        assert storage.data_str == "[[1, 2], [3, 4]]"

    def test_from_list_float(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (2, 2)
        assert storage.device.kind == "gpu"
        assert storage.data_str == "[[1.0, 2.0], [3.0, 4.0]]"

    def test_empty_list(self):
        data = [[]]
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (1, 0)
        assert storage.device.kind == "gpu"
        assert storage.data_str == "[[]]"

    def test_infer_rank(self):
        arr = [1.0, 2.0, 3.0]
        t = tp.Tensor(arr)
        assert t.trace_tensor.rank == 1
