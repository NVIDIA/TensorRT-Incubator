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

import pytest

import mlir_tensorrt.runtime.api as runtime
import tripy as tp

from mlir_tensorrt.compiler import ir

from tripy.backend.mlir import utils as mlir_utils
from tripy.common import Array
from tripy.common.datatype import DATA_TYPES
from tripy.flat_ir.flat_ir import FlatIR
from tripy.frontend.trace.ops import Storage
from tripy.frontend.trace.tensor import TraceTensor


class TestStorage:
    def test_cpu_storage(self):
        data = Array([1, 2, 3], shape=(3,), dtype=None, device=tp.device("cpu"))
        storage = Storage([], [], data)
        assert isinstance(storage.data.memref_value, runtime.MemRefValue)
        assert storage.data.memref_value.address_space == runtime.PointerType.host
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        data = Array([1, 2, 3], shape=(3,), dtype=None, device=tp.device("gpu"))
        storage = Storage([], [], data)
        assert isinstance(storage.data.memref_value, runtime.MemRefValue)
        assert storage.data.memref_value.address_space == runtime.PointerType.device
        assert storage.device.kind == "gpu"

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        # Given a int/float data list, store data with requested data type.
        if dtype not in {tp.float32, tp.int32, tp.int64}:
            pytest.skip(f"List to tp.Array conversion only supports float32, int32, int64. Got {dtype}")
        arr = [1.0, 2.0, 3.0] if dtype == tp.float32 else [1, 2, 3]
        data = Array(arr, shape=(3,), dtype=dtype, device=None)
        storage = Storage([], [], data)
        assert storage.dtype == dtype
        assert storage.dtype.name == dtype.name
        assert storage.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_mlir_conversion(self, dtype):
        # Given a int/float data list, store data with requested data type.
        if dtype not in {tp.float32, tp.int32, tp.int64}:
            pytest.skip(f"List to tp.Array conversion only supports float32, int32, int64. Got {dtype}")
        arr = [1.0, 2.0, 3.0] if dtype == tp.float32 else [1, 2, 3]
        data = Array(arr, shape=(3,), dtype=dtype, device=None)
        storage = Storage([], [TraceTensor("t0", None, dtype, None, None, 1, shape=[3])], data)
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            flat_ir = FlatIR()
            fir_outputs = [out.to_flat_ir() for out in storage.outputs]
            storage.to_flat_ir([], fir_outputs)
            flat_ir.integrate_subgraph([], fir_outputs)
            mlir_outputs = flat_ir.ops[0].to_mlir(operands=[])
            assert mlir_outputs[0].value.type.element_type == mlir_utils.get_mlir_dtype(dtype)

    def test_infer_rank(self):
        arr = [1.0, 2.0, 3.0]
        t = tp.Tensor(arr)
        assert t.trace_tensor.rank == 1
