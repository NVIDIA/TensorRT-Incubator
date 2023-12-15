from typing import Any, Optional, List, Tuple, Union
import numpy as np
import cupy as cp

import tripy.common.datatype
from tripy import util
from tripy.common.datatype import convert_numpy_to_tripy_dtype, convert_tripy_to_numpy_dtype
from tripy.common.device import Device


# The class abstracts away implementation differences between Torch, Jax, Cupy, NumPy, and List.
# Data is stored as a byte buffer, enabling interoperability across array libraries.
# The byte buffer is created using the `_convert_to_byte_buffer` function.
# Views with different data types can be created using the `view` method.
class Array:
    """
    A versatile array container that works with Torch, Jax, Cupy, NumPy, and List implementations.
    It can be used to store any object implementing dlpack interface.

    Example:
        from tripy.common.array import Array
        arr = Array([1, 2, 3], dtype=tripy.common.datatype.int32, device=Device("cpu"))
        assert arr.cpu_view(tripy.common.datatype.int32) == [1, 2, 3]

        arr = Array(numpy.array([1, 2, 3], dtype=np.float32), device=Device("cpu"))
        assert arr.cpu_view(np.float32) == np.array([1, 2, 3], np.float32)).all()
    """

    def __init__(
        self,
        data: Union[List, np.ndarray, cp.ndarray, "torch.Tensor", "jnp.ndarray"],
        dtype: tripy.common.datatype,
        shape: Optional[Tuple[int]],
        device: Device,
    ) -> None:
        """
        Initialize an Array object.

        Args:
            data: Input data list or an object implementing dlpack interface such as np.ndarray, cp.ndarray, torch.Tensor, or jnp.ndarray.
            dtype: Data type of the array.
            shape: Shape information for static allocation.
            device: Target device (tripy.device.Device("cpu") or tripy.device.Device("gpu")).
        """
        from tripy.frontend.dim import Dim

        assert dtype is None or dtype.__name__ in tripy.common.datatype.DATA_TYPES, "Invalid data type"

        self._module = np if device.kind == "cpu" else cp

        data_dtype = util.default(dtype, tripy.common.datatype.float32)
        if data is not None:
            data_dtype = convert_numpy_to_tripy_dtype(
                type(data[0]) if isinstance(data, List) and len(data) > 0 else data.dtype
            )
        assert dtype is None or data_dtype == dtype, f"{data_dtype} vs {dtype}"  # No Cast is supported.

        self.device: Device = device

        static_shape = None
        if shape is not None:
            static_shape = util.make_tuple([s.max if isinstance(s, Dim) else s for s in util.make_list(shape)])

        # Allocate dummy data
        if data is None:
            assert shape is not None
            data = self._module.empty(dtype=convert_tripy_to_numpy_dtype(data_dtype), shape=static_shape)
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

        # Data type of the array.
        self.dtype = data_dtype

    def cpu_view(self, dtype: Union[tripy.common.datatype.DataType, np.dtype]):
        """
        Create a cpu view of the array with a different data type.

        Args:
            dtype (tripy.common.datatype.DataType): Target data type for the view.

        Returns:
            np.ndarray: Numpy array view with the specified data type.
        """
        assert dtype is not None
        if not issubclass(dtype, np.floating) and not issubclass(dtype, np.integer):
            assert dtype.name in tripy.common.datatype.DATA_TYPES
            dtype = convert_tripy_to_numpy_dtype(dtype)

        if self.device.kind == "gpu":
            # Copy data from gpu to cpu.
            return self.byte_buffer.get().view(dtype)
        else:
            return self.byte_buffer.view(dtype)

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
    data: Union[List, np.ndarray, cp.ndarray, "torch.Tensor", "jnp.ndarray"],
    dtype: tripy.common.datatype.DataType,
    device: str,
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
    assert device in ["cpu", "gpu"]

    # Choose the appropriate module (NumPy or Cupy) based on the device
    _module = cp if device == "gpu" else np

    if isinstance(data, list):
        # Use array method with dtype to convert list to NumPy or Cupy array
        return _module.array(data, dtype=convert_tripy_to_numpy_dtype(dtype)).view(_module.uint8)
    else:
        # Use array method to convert data to NumPy or Cupy array
        return _module.array(data).view(_module.uint8)
