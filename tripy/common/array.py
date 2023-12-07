import cupy as cp
import numpy as np
from typing import List

import tripy.common.datatype

from tripy.common.device import device
from tripy.common.datatype import DataTypeConverter


class Array:
    def __init__(self, data: List[int] or List[float], dtype: tripy.common.datatype, device: device):
        """
        Initialize an Array object.

        Args:
            data (List[int] or List[float]): Input data as a list of integers or floats.
            dtype (tripy.common.datatype): Data type of the array (int32 or float32).
            device (tripy.common.device.device): Target device (cpu or gpu).
        """
        assert dtype in {tripy.common.datatype.int32, tripy.common.datatype.float32}, "Invalid data type"
        self._module = np if device.kind == "cpu" else cp

        # Convert input data to a numpy array with the specified dtype and store it as a byte buffer.
        self.byte_buffer = self._module.array(
            data, dtype=np.float32 if dtype == tripy.common.datatype.float32 else np.int32
        ).view(np.uint8)
        self.device = device

    def view(self, dtype: tripy.common.datatype):
        """
        Create a view of the array with a different data type.

        Args:
            dtype (tripy.common.datatype): Target data type for the view.

        Returns:
            np.ndarray: Numpy array view with the specified data type.
        """
        if self.device.kind == "gpu":
            return np.ascontiguousarray(self.byte_buffer.get()).view(
                DataTypeConverter.convert_tripy_to_numpy_dtype(dtype)
            )
        else:
            return np.ascontiguousarray(self.byte_buffer).view(DataTypeConverter.convert_tripy_to_numpy_dtype(dtype))

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
