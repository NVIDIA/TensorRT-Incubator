import cupy as cp
import numpy as np
import pytest
from mlir import ir

import tripy
from tripy.backend.mlir import utils as mlir_utils
from tripy.common.datatype import DATA_TYPES
from tripy.frontend.dim import Dim
from tripy.frontend.ops import Storage


class TestStorage:
    def test_cpu_storage(self):
        storage = Storage([], [], False, [1, 2, 3], shape=(Dim(3),), device=tripy.device("cpu"))
        assert isinstance(storage.data.byte_buffer, np.ndarray)
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        storage = Storage([], [], False, [1, 2, 3], shape=(Dim(3),), device=tripy.device("gpu"))
        assert isinstance(storage.data.byte_buffer, cp.ndarray)
        assert storage.device.kind == "gpu"

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.

        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        if dtype in {tripy.int8, tripy.int64, tripy.float16, tripy.uint8, tripy.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        data = [1, 2, 3] if dtype == tripy.int32 else [1.0, 2.0, 3.0]
        storage = Storage([], [], False, data, shape=(Dim(3),), dtype=dtype)
        assert storage.dtype == dtype
        assert storage.dtype.name == dtype.name
        assert storage.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_mlir_conversion(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.

        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        if dtype in {tripy.int8, tripy.int64, tripy.float16, tripy.uint8, tripy.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        data = [1, 2, 3] if dtype == tripy.int32 else [1.0, 2.0, 3.0]
        storage = Storage([], [], False, data, shape=(Dim(3),), dtype=dtype)
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            from tripy.flat_ir.flat_ir import FlatIR
            from tripy.flat_ir.tensor import FIRTensor
            from tripy.frontend.trace.tensor import TraceTensor

            flat_ir = FlatIR()
            out_tensor = TraceTensor("t0", None, [3], None, storage.dtype, None)
            storage.to_flat_ir(flat_ir, [], [FIRTensor(out_tensor)])
            outputs = flat_ir.ops[0].to_mlir(operands=[])
            assert outputs[0].value.type.element_type == mlir_utils.get_mlir_dtype(dtype)
