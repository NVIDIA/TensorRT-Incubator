from typing import Any, Optional, List, Tuple, Union
import numpy as np
import cupy as cp
import jax.numpy as jnp
import jax
import torch
import jaxlib
from tripy import util
import tripy.common.datatype
from tripy.common.datatype import DataTypeConverter
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
            data_dtype = DataTypeConverter.convert_numpy_to_tripy_dtype(
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
            data = np.zeros(dtype=DataTypeConverter.convert_tripy_to_numpy_dtype(data_dtype), shape=static_shape)

        # Convert input data to a byte buffer using DlpackConverter
        converter: DlpackConverter = DlpackConverter(self.device.kind)
        self.byte_buffer: Union[np.ndarray, cp.ndarray] = converter.convert(data, data_dtype)

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


class DlpackConverter:
    def __init__(self, target_device: str) -> None:
        """
        Initialize a DlpackConverter.

        Args:
            target_device (str): Target device ("cpu" or "gpu").
        """
        self.target_device: str = target_device
        self.converters: dict = {
            np.ndarray: self._convert,
            cp.ndarray: self._convert,
            torch.Tensor: self._convert,
            jaxlib.xla_extension.ArrayImpl: self._convert,
            list: self._convert,
        }

    def convert(self, data: Any, dtype: Any) -> Union[np.ndarray, cp.ndarray]:
        """
        Convert input data to a byte buffer.

        Args:
            data: Input data.
            dtype: Data type.

        Returns:
            np.ndarray or cp.ndarray: Byte buffer containing converted data.
        """
        converter: Any = self.converters.get(type(data))
        if converter is None:
            raise ValueError("Unsupported input type")
        return converter(data, dtype)

    def _convert(self, data: Any, dtype: Any) -> Union[np.ndarray, cp.ndarray]:
        """
        Common conversion logic for both CPU and GPU.

        Args:
            data: Input data.
            dtype: Data type.

        Returns:
            np.ndarray or cp.ndarray: Byte buffer containing converted data.
        """
        assert self.target_device in ["cpu", "gpu"]
        move_fn: Any = self._to_cpu if self.target_device == "cpu" else self._to_gpu
        return move_fn(data, dtype)

    def _to_cpu(self, data: Any, dtype: Any) -> np.ndarray:
        """
        Convert input data to a byte buffer on CPU.

        Args:
            data: Input data.
            dtype: Data type.

        Returns:
            np.ndarray: Byte buffer containing converted data.
        """
        assert self.target_device == "cpu"
        if isinstance(data, torch.Tensor):
            data = self._move_to_target_device(data, self.target_device)
            return np.array(torch.utils.dlpack.from_dlpack(data)).view(np.uint8)
        if isinstance(data, jnp.ndarray):
            data = self._move_to_target_device(data, self.target_device)
            return np.array(jax.dlpack.from_dlpack(data)).view(np.uint8)
        elif isinstance(data, list):
            return np.array(data, dtype=DataTypeConverter.convert_tripy_to_numpy_dtype(dtype)).view(np.uint8)
        elif isinstance(data, cp.ndarray):
            return np.array(data.get()).view(np.uint8)
        elif isinstance(data, np.ndarray):
            return np.array(data).view(np.uint8)
        else:
            raise ValueError("Unsupported input type")

    def _to_gpu(self, data: Any, dtype: Any) -> cp.ndarray:
        """
        Convert input data to a byte buffer on GPU.

        Args:
            data: Input data.
            dtype: Data type.

        Returns:
            cp.ndarray: Byte buffer containing converted data.
        """
        assert self.target_device == "gpu"
        if isinstance(data, torch.Tensor):
            data = self._move_to_target_device(data, self.target_device)
            return cp.array(torch.utils.dlpack.from_dlpack(data)).view(np.uint8)
        if isinstance(data, jnp.ndarray):
            data = self._move_to_target_device(data, self.target_device)
            return cp.array(jax.dlpack.from_dlpack(data)).view(np.uint8)
        elif isinstance(data, list):
            return cp.array(data, dtype=DataTypeConverter.convert_tripy_to_numpy_dtype(dtype)).view(cp.uint8)
        elif isinstance(data, (np.ndarray, cp.ndarray)):
            return cp.array(data).view(cp.uint8)
        else:
            raise ValueError("Unsupported input type")

    def _move_to_target_device(self, data: Any, target_device: str) -> Any:
        """
        Move input data to the target device.

        Args:
            data: Input data.
            target_device: Target device.

        Returns:
            data: Data moved to the target device.
        """
        if isinstance(data, torch.Tensor):
            if target_device not in str(data.device).lower():
                if target_device == "gpu":
                    target_device = "cuda"
                data = data.to(target_device)
        elif isinstance(data, jnp.ndarray):
            if target_device not in str(jax.devices(target_device)[0]).lower():
                data = jax.device_put(data, jax.devices(target_device)[0])
        else:
            raise ValueError("Unsupported input type")

        return data
