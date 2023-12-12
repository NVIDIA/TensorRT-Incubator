from typing import Any, Optional, List, Tuple, Union
import numpy as np
import cupy as cp
import jax.numpy as jnp
import jax
import torch
from tripy import util
import tripy.common.datatype
from tripy.common.datatype import convert_numpy_to_tripy_dtype, convert_tripy_to_numpy_dtype
from tripy.common.device import Device


class Array:
    def __init__(
        self,
        data: Union[list, np.ndarray, cp.ndarray, torch.Tensor, jnp.ndarray],
        dtype: tripy.common.datatype,
        shape: Optional[Tuple[int]],
        device: Device,
    ) -> None:
        """
        Initialize an Array object.

        Args:
            data (list or np.ndarray or cp.ndarray or torch.Tensor or jnp.ndarray): Input data.
            dtype: Data type of the array.
            device (str): Target device ("cpu" or "gpu").
        """
        from tripy.frontend.dim import Dim

        assert dtype is None or dtype.__name__ in tripy.common.datatype.DATA_TYPES, "Invalid data type"

        self._module = np if device.kind == "cpu" else cp

        data_dtype = util.default(dtype, tripy.common.datatype.float32)
        if data is not None:
            data_dtype = convert_numpy_to_tripy_dtype(
                type(data[0]) if isinstance(data, List) and len(data) > 0 else data.dtype
            )
        assert dtype is None or data_dtype == dtype  # No Cast is supported.

        self.device: Device = device

        static_shape = None
        if shape is not None:
            static_shape = util.make_tuple([s.max if isinstance(s, Dim) else s for s in util.make_list(shape)])

        # Allocate dummy data
        if data is None:
            assert shape is not None
            data = np.zeros(dtype=convert_tripy_to_numpy_dtype(data_dtype), shape=static_shape)

        # Convert input data to a byte buffer.
        self.byte_buffer: Union[np.ndarray, cp.ndarray] = _convert_to_byte_buffer(data, data_dtype, self.device.kind)

        # Figure out how to extract shape information? Does it work for jax and torch?
        self.shape = (
            static_shape
            if static_shape is not None
            else util.make_tuple(len(data))
            if isinstance(data, List)
            else data.shape
        )

        # Store dtype
        self.dtype = data_dtype

    def view(self, dtype: tripy.common.datatype):
        """
        Create a view of the array with a different data type.

        Args:
            dtype (tripy.common.datatype): Target data type for the view.

        Returns:
            np.ndarray: Numpy array view with the specified data type.
        """
        if self.device.kind == "gpu":
            return np.ascontiguousarray(self.byte_buffer.get()).view(convert_tripy_to_numpy_dtype(dtype))
        else:
            return np.ascontiguousarray(self.byte_buffer).view(convert_tripy_to_numpy_dtype(dtype))

    def __eq__(self, other) -> bool:
        """
        Check if two arrays are equal.

        Args:
            other (Array): Another Array object for comparison.

        Returns:
            bool: True if the arrays are equal, False otherwise.
        """
        if self._module != other._module:
            return False
        return self._module.array_equal(self.byte_buffer, other.byte_buffer)


def _convert_to_byte_buffer(
    data: Union[List, np.ndarray, cp.ndarray, torch.Tensor, jnp.ndarray], dtype: tripy.common.datatype, device: str
) -> Union[np.ndarray, cp.ndarray]:
    """
    Common conversion logic for both CPU and GPU.

    Args:
        data: Input data.
        dtype: Data type.
        device (str): Target device ("cpu" or "gpu").

    Returns:
        np.ndarray or cp.ndarray: Byte buffer containing converted data.
    """

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

    assert device in ["cpu", "gpu"]

    # Choose the appropriate module (NumPy or Cupy) based on the device
    _module = cp if device == "gpu" else np
    data = _move_to_device(data, device)

    if isinstance(data, list):
        # Use array method with dtype to convert list to NumPy or Cupy array
        return _module.array(data, dtype=convert_tripy_to_numpy_dtype(dtype)).view(_module.uint8)
    else:
        # Use array method to convert data to NumPy or Cupy array
        return _module.array(data).view(_module.uint8)
