from typing import Any, List

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import mlir_tensorrt.runtime.api as runtime
import tripy as tp

from tests.helper import NUMPY_TYPES, torch_type_supported
from tripy import utils
from tripy.common.array import Array
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype
from tripy.backend.mlir.utils import convert_tripy_dtype_to_runtime_dtype

data_list = []

# Create a data list for NumPy arrays
np_data = [np.ones(1, dtype=dtype) for dtype in NUMPY_TYPES]
data_list.extend(np_data)

# Extend the data list for Cupy arrays
data_list.extend([cp.array(data) for data in np_data])

# Extend the data list for Torch CPU tensors
data_list.extend([torch.tensor(data) for data in filter(torch_type_supported, np_data)])

# Extend the data list for Torch GPU tensors
data_list.extend([torch.tensor(data).to(torch.device("cuda")) for data in filter(torch_type_supported, np_data)])

# Extend the data list for Jax CPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cpu")[0]) for data in np_data])

# Extend the data list for Jax GPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cuda")[0]) for data in np_data])


class TestArray:

    @pytest.mark.parametrize("input_data", data_list, ids=lambda data: f"{type(data).__qualname__}")
    def test_creation(self, input_data):
        """
        Test the creation of Array objects with different devices and data types.
        """
        dtype = convert_frontend_dtype_to_tripy_dtype(input_data.dtype)
        shape = input_data.shape
        if dtype is not None:
            arr = Array(input_data, dtype, shape)
            assert isinstance(arr, Array)
            assert isinstance(arr.memref_value, runtime.MemRefValue)
            assert arr.memref_value.dtype == convert_tripy_dtype_to_runtime_dtype(dtype)
            assert (
                arr.memref_value.address_space == runtime.PointerType.host
                if arr.device.kind == "cpu"
                else runtime.PointerType.device
            )

    @pytest.mark.parametrize("np_dtype", NUMPY_TYPES)
    def test_0d(self, np_dtype):
        dtype = convert_frontend_dtype_to_tripy_dtype(np_dtype)
        arr = Array(np.array(1, dtype=np_dtype), dtype, None, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == tuple()
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_nested_list(self, dtype):
        arr = Array([[1, 2], [3, 4]], dtype, None, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (2, 2)
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_empty_dimension(self, dtype):
        arr = Array([[], [], []], dtype=dtype, shape=(3, 0), device=tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (
            3,
            0,
        )
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_later_empty_dimensions(self, dtype):
        # consistent to call it (3, 0, ...) because it can't be checked further
        arr = Array([[], [], []], dtype=dtype, shape=(3, 0, 1, 2), device=tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (3, 0, 1, 2)
        assert arr.dtype == dtype

    def test_missing_data_shape(self):
        with pytest.raises(tp.TripyException, match="Shape must be provided when data is None.") as exc:
            _ = Array(None, tp.float32, None, tp.device("cpu"))
        print(str(exc.value))

    def test_missing_data_dtype(self):
        with pytest.raises(tp.TripyException, match="Datatype must be provided when data is None.") as exc:
            _ = Array(None, None, (), tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_dtype(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect dtype.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), tp.float32, None, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), None, (3,), tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape_list(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array((1, 2, 3), None, (5,), tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape_nested_list(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array([[1, 2, 3], [4, 5, 6]], None, (1, 2, 3), tp.device("cpu"))
        print(str(exc.value))

    def test_inconsistent_jagged_list(self):
        with pytest.raises(tp.TripyException, match="Length of list at index 1 does not match at dimension 1") as exc:
            _ = Array([[1, 2, 3], [4, 5]], None, (2, 3), tp.device("cpu"))
        print(str(exc.value))

    def test_inconsistent_scalar_value(self):
        with pytest.raises(tp.TripyException, match="Expected sequence at index 1, got int") as exc:
            _ = Array([[1, 2, 3], 0], None, (2, 3), tp.device("cpu"))
        print(str(exc.value))

    def test_unsupported_list_element(self):
        from decimal import Decimal

        with pytest.raises(tp.TripyException, match="List element type can only be int or float.") as exc:
            _ = Array([Decimal(0)], None, None, tp.device("cpu"))
        print(str(exc.value))

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_supported_array_type(self, dtype):
        arr = Array([0], dtype=dtype, shape=None, device=tp.device("cpu"))
        assert isinstance(arr.memref_value, runtime.MemRefValue)
        assert arr.memref_value.dtype == convert_tripy_dtype_to_runtime_dtype(dtype)
        assert arr.memref_value.address_space == runtime.PointerType.host

    @pytest.mark.parametrize("dtype", [tp.float16, tp.float8, tp.int8, tp.int4])
    def test_unsupported_array_type(self, dtype):
        with pytest.raises(
            tp.TripyException,
            match=f"Tripy array from list can be constructed with float32, int32, or int64, got {dtype}",
        ) as exc:
            _ = Array([0], dtype=dtype, shape=None, device=tp.device("cpu"))
        print(str(exc.value))
