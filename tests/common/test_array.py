from typing import List

import cupy as cp
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tripy.common.array import Array
from tripy.common.device import Device
from tripy.common.datatype import DataTypeConverter


def torch_type_supported(data: np.ndarray):
    unsupported_dtypes = [np.uint16, np.uint32, np.uint64]
    return data.dtype not in unsupported_dtypes


# Supported NumPy data types
numpy_dtypes = [
    np.float16,
    np.float32,
    np.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

# Create NumPy input data list.
np_data = [np.ones(1, dtype=dtype) for dtype in numpy_dtypes]
data_list = []

# Create a data list for NumPy arrays
data_list.extend(np_data)

# Extend the data list for Cupy arrays
data_list.extend([cp.array(data) for data in np_data])

# Extend the data list for Torch CPU tensors
data_list.extend([torch.tensor(data) for data in list(filter(torch_type_supported, np_data))])

# Extend the data list for Torch GPU tensors
data_list.extend([torch.tensor(data).to(torch.device("cuda")) for data in list(filter(torch_type_supported, np_data))])

# Extend the data list for Jax arrays
data_list.extend([jnp.array(data) for data in np_data])

# Define parameters for device type and index
device_params = [
    {"device_type": "cpu", "device_index": None},
    {"device_type": "gpu", "device_index": 0},
]


@pytest.mark.parametrize("device_param", device_params)
@pytest.mark.parametrize("input_data", data_list)
def test_array_creation(device_param, input_data):
    """
    Test the creation of Array objects with different devices and data types.
    """
    device_type = device_param["device_type"]
    device_index = device_param["device_index"]
    device = Device(device_type, device_index)
    dtype = DataTypeConverter.convert_numpy_to_tripy_dtype(input_data.dtype)
    shape = (len(List),) if isinstance(input_data, List) else input_data.shape
    if dtype is not None:
        arr = Array(input_data, dtype, shape, device)

        assert isinstance(arr, Array)
        assert isinstance(arr.byte_buffer, (np.ndarray, cp.ndarray))
        assert arr.byte_buffer.dtype == np.uint8 or arr.byte_buffer.dtype == cp.uint8
        assert arr.device.kind == device_type
        assert arr.device.index == device_index
