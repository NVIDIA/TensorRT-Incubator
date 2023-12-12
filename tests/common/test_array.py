from typing import List

import cupy as cp
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import jax

from tripy.common.array import Array
from tripy.common.device import Device
from tripy.common.datatype import convert_numpy_to_tripy_dtype

from tests.helper import torch_type_supported
from tests.helper import NUMPY_TYPES

# Create NumPy input data list.
np_data = [np.ones(1, dtype=dtype) for dtype in NUMPY_TYPES]
data_list = []

# Create a data list for NumPy arrays
data_list.extend(np_data)

# Extend the data list for Cupy arrays
data_list.extend([cp.array(data) for data in np_data])

# Extend the data list for Torch CPU tensors
data_list.extend([torch.tensor(data) for data in list(filter(torch_type_supported, np_data))])

# Extend the data list for Torch GPU tensors
data_list.extend([torch.tensor(data).to(torch.device("cuda")) for data in list(filter(torch_type_supported, np_data))])

# Extend the data list for Jax CPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cpu")[0]) for data in np_data])

# Extend the data list for Jax GPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cuda")[0]) for data in np_data])

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
    dtype = convert_numpy_to_tripy_dtype(input_data.dtype)
    shape = (len(List),) if isinstance(input_data, List) else input_data.shape
    if dtype is not None:
        arr = Array(input_data, dtype, shape, device)

        assert isinstance(arr, Array)
        assert isinstance(arr.byte_buffer, (np.ndarray, cp.ndarray))
        assert arr.byte_buffer.dtype == np.uint8 or arr.byte_buffer.dtype == cp.uint8
        assert arr.device.kind == device_type
        assert arr.device.index == device_index
