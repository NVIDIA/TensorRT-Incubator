from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from tripy import util
from tripy.common.datatype import convert_numpy_to_tripy_dtype, convert_tripy_to_numpy_dtype, DATA_TYPES
from tripy.common.device import device


# The class abstracts away implementation differences between Torch, Jax, Cupy, NumPy, and List.
# Data is stored as a byte buffer, enabling interoperability across array libraries.
# The byte buffer is created using the `_convert_to_byte_buffer` function.
# Views with different data types can be created using the `view` method.
class Array:
    """
    A versatile array container that works with Torch, Jax, Cupy, NumPy, and List implementations.
    It can be used to store any object implementing dlpack interface.
    """

    def __init__(
        self,
        data: Union[List, np.ndarray, cp.ndarray, "torch.Tensor", "jnp.ndarray"],
        dtype: "tripy.dtype",
        shape: Optional[Tuple[int]],
        device: device,
    ) -> None:
        """
        Initialize an Array object.

        Args:
            data: Input data list or an object implementing dlpack interface such as np.ndarray, cp.ndarray, torch.Tensor, or jnp.ndarray.
            dtype: Data type of the array.
            shape: Shape information for static allocation.
            device: Target device (tripy.Device("cpu") or tripy.Device("gpu")).
        """
        from tripy.frontend.dim import Dim
        import tripy.common.datatype

        assert dtype is None or dtype.__name__ in tripy.common.datatype.DATA_TYPES, "Invalid data type"

        self._module = np if device.kind == "cpu" else cp

        data_dtype = util.default(dtype, tripy.common.datatype.float32)
        if data is not None:
            data_dtype = convert_numpy_to_tripy_dtype(
                type(data[0]) if isinstance(data, List) and len(data) > 0 else data.dtype
            )
        assert (
            dtype is None or data_dtype == dtype
        ), f"Incorrect data type. Note: Input data had type: {data_dtype} but provided dtype was: {dtype}"  # No Cast is supported.

        self.device: device = device

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

    def view(self):
        """
        Create a NumPy Or CuPy array of underlying datatype.
        """
        assert self.dtype is not None
        assert self.dtype.name in DATA_TYPES
        dtype = convert_tripy_to_numpy_dtype(self.dtype)
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
    dtype: "tripy.dtype",
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
        if not data.shape:
            # Numpy requires reshaping to 1d because of the following error:
            #    "ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged"
            data = data.reshape(
                1,
            )

        return _module.array(data).view(_module.uint8)
