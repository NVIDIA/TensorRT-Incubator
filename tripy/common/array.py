from typing import List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from tripy import utils
from tripy.common.datatype import convert_numpy_to_tripy_dtype, convert_tripy_to_numpy_dtype, DATA_TYPES
from tripy.common.exception import raise_error
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
        shape: Optional[Tuple["Dim"]],
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
        import tripy.common.datatype

        assert dtype is None or dtype.__name__ in tripy.common.datatype.DATA_TYPES, "Invalid data type"

        self._module = np if device.kind == "cpu" else cp
        self.device = device

        data_dtype = utils.default(dtype, tripy.common.datatype.float32)
        static_shape = (
            utils.make_tuple([s.runtime_value for s in utils.make_list(shape)]) if shape is not None else None
        )
        assert shape is None or all(s > 0 for s in static_shape)

        if data is None:
            if shape is None:
                raise_error("Shape must be provided when data is None.", [])
            self.shape = static_shape
            self.dtype = data_dtype
            # Allocate dummy data
            data = self._module.empty(dtype=convert_tripy_to_numpy_dtype(data_dtype), shape=self.shape)
        else:
            if isinstance(data, (List, int, float, tuple)):
                data = self._module.array(data, dtype=self._get_element_type(data))

            data_dtype = convert_numpy_to_tripy_dtype(data.dtype)
            if not data_dtype:
                raise_error(f"Data has unsupported dtype: {data.dtype}")
            if dtype is not None and data_dtype != dtype:
                # Check the consistency between data_dtype and dtype
                # No cast is supported during initialization
                raise_error(
                    "Data has incorrect dtype.",
                    details=[f"input data had type: {data_dtype}, ", f"but provided dtype was: {dtype}"],
                )
            self.dtype = data_dtype

            if shape is not None and static_shape != data.shape:
                # Check consistency between data.shape and shape
                # No reshape is supported during initialization
                raise_error(
                    "Data has incorrect shape.",
                    details=[f"input data had shape: {data.shape}, ", f"but provided shape was: {static_shape}"],
                )
            self.shape = data.shape
        # Convert data with correct dtype and shape to a byte buffer.
        self.byte_buffer: Union[np.ndarray, cp.ndarray] = self._convert_to_byte_buffer(data)

    def view(self):
        """
        Create a NumPy Or CuPy array of underlying datatype.
        """
        assert self.dtype is not None
        assert self.dtype.name in DATA_TYPES
        dtype = convert_tripy_to_numpy_dtype(self.dtype)
        out = self.byte_buffer.view(dtype)
        if not self.shape:
            # Reshape back to 0-D
            out = out.reshape(())
        return out

    def __str__(self):
        return str(self.view())

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

    def _convert_to_byte_buffer(self, data):
        """
        Converts data to byte buffer that works for both GPU and CPU.

        Args:
            data: Input data.

        Returns:
            np.ndarray or cp.ndarray: Byte buffer containing converted data.
        """
        if not data.shape:
            # Numpy requires reshaping to 1d because of the following error:
            #    "ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged"
            data = data.reshape((1,))
        return self._module.array(data).view(self._module.uint8)

    def _get_element_type(self, elements):
        e = elements
        while isinstance(e, List) or isinstance(e, tuple):
            e = e[0]
        if isinstance(e, int):
            return self._module.int32
        elif isinstance(e, float):
            return self._module.float32
        else:
            raise_error(
                "Unsupported element type.",
                details=[f"List element type can only be int or float.", f"Got element {e} of type {type(e)}."],
            )
