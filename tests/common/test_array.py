from typing import Any, List

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import tripy as tp
from tests.helper import NUMPY_TYPES, torch_type_supported
from tripy import utils
from tripy.common.array import Array, convert_to_tripy_dtype

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

# Define parameters for device type and index
device_params = [
    {"device_type": "cpu", "device_index": None},
    {"device_type": "gpu", "device_index": 0},
]


def _move_to_device(data: Any, device: str) -> Any:
    """Move input data to the target device."""
    if isinstance(data, torch.Tensor):
        # Use torch's to method to move data to the target device
        if device not in str(data.device).lower():
            device = "cuda" if device == "gpu" else device
            data = data.to(device)
    elif isinstance(data, jnp.ndarray):
        # Use jax's device_put method to move data to the target device
        if device not in str(jax.devices(device)[0]).lower():
            data = jax.device_put(data, jax.devices(device)[0])
    elif isinstance(data, cp.ndarray):
        # Use Cupy's get method to move data to CPU
        if device == "cpu":
            data = data.get()
    else:
        # Ensure that data is either a NumPy array or a list
        assert isinstance(data, (np.ndarray, List))

    return data


class TestArray:

    @pytest.mark.parametrize(
        "device_param", device_params, ids=lambda param: f"{param['device_type']}:{param['device_index']}"
    )
    @pytest.mark.parametrize("input_data", data_list, ids=lambda data: f"{type(data).__qualname__}")
    def test_creation(self, device_param, input_data):
        """
        Test the creation of Array objects with different devices and data types.
        """
        device_type = device_param["device_type"]
        device_index = device_param["device_index"]
        device = tp.device(f"{device_type}:{device_index}" if device_index is not None else device_type)
        dtype = convert_to_tripy_dtype(input_data.dtype)
        shape = input_data.shape
        if dtype is not None:
            arr = Array(_move_to_device(input_data, device_type), dtype, shape, device)
            assert isinstance(arr, Array)
            assert isinstance(arr.byte_buffer, (np.ndarray, cp.ndarray))
            assert arr.byte_buffer.dtype == np.uint8 or arr.byte_buffer.dtype == cp.uint8
            assert arr.device.kind == device_type
            assert arr.device.index == utils.default(device_index, 0)

    @pytest.mark.parametrize("np_dtype", NUMPY_TYPES)
    def test_0d(self, np_dtype):
        dtype = convert_to_tripy_dtype(np_dtype)
        arr = Array(np.array(1, dtype=np_dtype), dtype, None, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == tuple()
        assert arr.dtype == dtype

    def test_nested_list(self):
        dtype = tp.int32
        arr = Array([[1, 2], [3, 4]], dtype, None, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (2, 2)
        assert arr.dtype == dtype

    def test_missing_data_shape(self):
        with pytest.raises(tp.TripyException, match="Shape must be provided when data is None.") as exc:
            _ = Array(None, tp.float32, None, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_dtype(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect dtype.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), tp.float32, None, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), None, (3,), tp.device("cpu"))
        print(str(exc.value))

    def test_unsupported_list_element(self):
        from decimal import Decimal

        with pytest.raises(tp.TripyException, match="List element type can only be int or float.") as exc:
            _ = Array([Decimal(0)], None, None, tp.device("cpu"))
        print(str(exc.value))
