import cupy as cp
import numpy as np
import pytest
from mlir_tensorrt.compiler import ir

import tripy as tp
from tripy.backend.mlir import utils as mlir_utils
from tripy.common import Array
from tripy.common.datatype import DATA_TYPES
from tripy.flat_ir.flat_ir import FlatIR
from tripy.frontend.trace.ops import Storage
from tripy.frontend.trace.tensor import TraceTensor


class TestStorage:
    def test_cpu_storage(self):
        data = Array([1, 2, 3], dtype=None, shape=(3,), device=tp.device("cpu"))
        storage = Storage([], [], data)
        assert isinstance(storage.data.byte_buffer, np.ndarray)
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        data = Array([1, 2, 3], dtype=None, shape=(3,), device=tp.device("gpu"))
        storage = Storage([], [], data)
        assert isinstance(storage.data.byte_buffer, cp.ndarray)
        assert storage.device.kind == "gpu"

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.

        if dtype in {tp.int4, tp.bfloat16, tp.float8}:
            pytest.skip("Type is not supported by numpy/cupy")

        if dtype in {tp.int8, tp.int64, tp.float16, tp.uint8, tp.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        arr = [1, 2, 3] if dtype == tp.int32 else [1.0, 2.0, 3.0]
        data = Array(arr, shape=(3,), dtype=dtype, device=None)
        storage = Storage([], [], data)
        assert storage.dtype == dtype
        assert storage.dtype.name == dtype.name
        assert storage.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_mlir_conversion(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.

        if dtype in {tp.int4, tp.bfloat16, tp.float8}:
            pytest.skip("Type is not supported by numpy/cupy")

        if dtype in {tp.int8, tp.int64, tp.float16, tp.uint8, tp.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        arr = [1, 2, 3] if dtype == tp.int32 else [1.0, 2.0, 3.0]
        data = Array(arr, shape=(3,), dtype=dtype, device=None)
        storage = Storage([], [TraceTensor("t0", None, [3], dtype, None, None, None)], data)
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
