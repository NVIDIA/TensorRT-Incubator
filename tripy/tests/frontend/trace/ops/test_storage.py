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

import tripy as tp


from tripy.backend.mlir import memref
from tripy.frontend.trace.ops import Storage
from tripy.frontend.trace.tensor import TraceTensor


class TestStorage:

    @pytest.mark.parametrize("device", ["cpu", "gpu"])
    def test_from_memref(self, device):
        module = np if device == "cpu" else cp
        data = memref.create_memref_view(module.ones((2, 2), dtype=module.float32))
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (2, 2)
        assert storage.device.kind == device

    def test_from_list(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data)
        assert storage.dtype == tp.float32
        assert storage.shape == (2, 2)
        assert storage.device.kind == "gpu"

    def test_empty_list(self):
        data = [[]]
        storage = Storage([], [TraceTensor("test", None, None, None, None, None)], data, dtype=tp.float16)
        assert storage.dtype == tp.float16
        assert storage.shape == (1, 0)
        assert storage.device.kind == "gpu"

    def test_infer_rank(self):
        arr = [1.0, 2.0, 3.0]
        t = tp.Tensor(arr)
        assert t.trace_tensor.rank == 1
