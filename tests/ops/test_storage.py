import cupy as cp
import numpy as np
import pytest
from mlir import ir

import tripy
from tripy.backend.mlir import utils as mlir_utils
from tripy.datatype import DATA_TYPES
from tripy.ops import Storage


class TestStorage:
    def test_cpu_storage(self):
        storage = Storage([1, 2, 3], device=tripy.device("cpu"))
        assert isinstance(storage.data, np.ndarray)
        assert storage.device.kind == "cpu"

    def test_gpu_storage(self):
        storage = Storage([1, 2, 3], device=tripy.device("gpu"))
        assert isinstance(storage.data, cp.ndarray)
        assert storage.device.kind == "gpu"

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        storage = Storage([1, 2, 3], dtype=dtype)
        assert storage.dtype == dtype
        assert storage.data.dtype.name == dtype.name
        assert storage.data.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_mlir_conversion(self, dtype):
        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        # TODO (pranavm): Figure out how to make boolean types work.
        if dtype in {tripy.bool}:
            pytest.skip("Bool is not working correctly yet")

        storage = Storage([1, 2, 3], dtype=dtype)
        with mlir_utils.make_ir_context(), ir.Location.unknown():
            outputs = storage.to_mlir(inputs=[])
            assert outputs[0].value.type.element_type == mlir_utils.convert_dtype(dtype)
