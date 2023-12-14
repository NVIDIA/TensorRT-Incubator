import cupy as cp
import numpy as np
import pytest
from mlir import ir

import tripy
from tripy.backend.mlir import utils as mlir_utils
from tripy.common.datatype import DATA_TYPES
from tripy.ops import Storage


class TestStorage:
    def test_cpu_storage(self):
        storage = Storage([1, 2, 3], shape=(3,), device=tripy.device("cpu"))
        assert isinstance(storage.data.byte_buffer, np.ndarray)
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        storage = Storage([1, 2, 3], shape=(3,), device=tripy.device("gpu"))
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
        storage = Storage(data, shape=(3,), dtype=dtype)
        assert storage.dtype == dtype
        assert storage.dtype.name == dtype.name
        assert storage.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_mlir_conversion(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.

        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        # TODO (#26): Figure out how to make boolean types work.
        if dtype in {tripy.bool}:
            pytest.skip("Bool is not working correctly yet")

        if dtype in {tripy.int8, tripy.int64, tripy.float16, tripy.uint8, tripy.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        data = [1, 2, 3] if dtype == tripy.int32 else [1.0, 2.0, 3.0]
        storage = Storage(data, shape=(3,), dtype=dtype)
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            outputs = storage.to_mlir(inputs=[])
            assert outputs[0].value.type.element_type == mlir_utils.get_mlir_dtype(dtype)
