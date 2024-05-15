from typing import Any, List, Optional, Tuple, Union
import array

from tripy import utils
from tripy.common.device import device as tp_device
from tripy.common.exception import raise_error
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype, is_supported_array_type, get_element_type

import mlir_tensorrt.runtime.api as runtime
import tripy.common.datatype


def check_data_consistency(data, shape, dtype):
    if dtype is not None:
        data_dtype = convert_frontend_dtype_to_tripy_dtype(data.dtype)
        if not data_dtype:
            raise_error(f"Data has unsupported dtype: {data.dtype}")
        if data_dtype != dtype:
            raise_error(
                "Data has incorrect dtype.",
                details=[f"Input data had type: {data_dtype}, ", f"but provided dtype was: {dtype}"],
            )
    if shape is not None and data.shape != shape:
        raise_error(
            "Data has incorrect shape.",
            details=[
                f"Input data had shape: {data.shape}, ",
                f"but provided runtime shape was: {shape}",
            ],
        )


# The class abstracts away implementation differences between Torch, Jax, Cupy, NumPy, and List.
# Data is stored as a byte buffer, enabling interoperability across array libraries.
# The byte buffer is created using the `convert_to_byte_buffer` function.
# Views with different data types can be created using the `view` method.
class Array:
    """
    A versatile array container that works with Torch, Jax, Cupy, NumPy, and List implementations.
    It can be used to store any object implementing dlpack interface.
    """

    def __init__(
        self,
        data: Union[List, "np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        dtype: "tripy.dtype",
        shape: Optional[Tuple[int]],
        device: tp_device,
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

        assert dtype is None or isinstance(dtype, tripy.common.datatype.dtype), "Invalid data type"
        assert shape is None or all(s >= 0 for s in shape)

        self.device = utils.default(device, tp_device("cpu"))

        if data is None:
            if dtype is None:
                raise_error("Datatype must be provided when data is None.", [])
            if shape is None:
                raise_error("Shape must be provided when data is None.", [])
            self.dtype = dtype
            self.shape = shape
        else:
            if isinstance(data, (List, int, float, tuple)):
                if dtype is None:
                    element_type = get_element_type(data)
                else:
                    element_type = convert_frontend_dtype_to_tripy_dtype(dtype)
                self.dtype = element_type
                if shape is None:
                    # Consider if a list is a nested list
                    self.shape = tuple(utils.get_shape(data))
                else:
                    self.shape = shape
            else:
                check_data_consistency(data, shape, dtype)
                self.dtype = convert_frontend_dtype_to_tripy_dtype(data.dtype)
                self.shape = data.shape

        # Store the memref_value
        self.runtime_client = runtime.RuntimeClient()
        self.data_ref = data  # Ensure that data does not go out of scope when we create a view over it.
        self.memref_value = self._memref(data)

    def data(self) -> List[Union[float, int]]:
        if self.memref_value.address_space == runtime.PointerType.device:
            host_memref = self.runtime_client.copy_to_host(
                device_memref=self.memref_value,
            )
            return memoryview(host_memref).tolist()
        return memoryview(self.memref_value).tolist()

    def _prettyprint(self, threshold=1000, linwidth=75):
        data = self.data()
        if isinstance(data, (float, int)):
            return str(data)

        # Limit printing elements. Defaults to numpy print options i.e. 1000.
        data_str = []
        for item in data[:threshold]:
            data_str.append(str(item))

        # Limit line width. Defaults to numpy print options i.e. 75.
        lines = []
        current_line = "["
        for item_str in data_str:
            if len(current_line) + len(item_str) + 2 > linwidth:
                lines.append(current_line.rstrip(", ") + ",")
                current_line = " " + item_str + ", "
            else:
                current_line += item_str + ", "
        if current_line:
            lines.append(current_line.rstrip(", ") + "]")

        return "\n".join(lines)

    def _memref(self, data):
        from tripy.backend.mlir.utils import convert_tripy_dtype_to_runtime_dtype

        mlirtrt_device = self.runtime_client.get_devices()[0] if self.device == tp_device("gpu") else None

        if data is None:
            # Allocate corresponding memref on host.
            return self.runtime_client.create_memref(
                shape=list(self.shape), dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype), device=mlirtrt_device
            )
        else:
            if isinstance(data, (List, tuple, int, float)):
                if not is_supported_array_type(self.dtype):
                    raise_error(
                        f"Tripy array from list can be constructed with float32, int32, or int64, got {self.dtype}"
                    )

                def _get_array_type_unicode(dtype: "tripy.common.datatype.dtype") -> str:
                    assert dtype is not None
                    assert is_supported_array_type(dtype)
                    unicode = {
                        tripy.common.datatype.int32: "i",
                        tripy.common.datatype.float32: "f",
                        tripy.common.datatype.int64: "q",
                    }
                    return unicode.get(dtype)

                return self.runtime_client.create_memref(
                    array.array(_get_array_type_unicode(self.dtype), utils.flatten_list(data)),
                    shape=list(self.shape),
                    dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                    device=mlirtrt_device,
                )
            else:
                # TODO(#182): Use DLPack/buffer protocol to convert FW types to MemRefValue.
                # Assume data is allocated using external framework. Create a view over it.
                from tripy.backend.mlir.utils import convert_tripy_dtype_to_runtime_dtype

                if self.device == tp_device("cpu"):
                    if hasattr(data, "__array_interface__"):
                        return self.runtime_client.create_host_memref_view(
                            int(data.ctypes.data),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                        )
                    elif hasattr(data, "data_ptr") and data.device.type == "cpu":
                        # Torch tensor allocated on CPU
                        return self.runtime_client.create_host_memref_view(
                            int(data.data_ptr()),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                        )
                    elif hasattr(data, "data_ptr") and data.device.type == "cuda":
                        # Torch tensor allocated on GPU
                        memref_value = self.runtime_client.create_device_memref_view(
                            int(data.data_ptr()),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            device=self.runtime_client.get_devices()[0],
                        )
                        return self.runtime_client.copy_to_host(
                            host_memref=memref_value,
                        )
                    elif hasattr(data, "data") and hasattr(data, "__cuda_array_interface__"):
                        memref_value = self.runtime_client.create_device_memref_view(
                            int(data.data.ptr),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            device=self.runtime_client.get_devices()[0],
                        )
                        return self.runtime_client.copy_to_host(
                            host_memref=memref_value,
                        )
                    elif hasattr(data, "__array__"):
                        arr = data.__array__()
                        if hasattr(arr, "__array_interface__"):
                            # Jax tensor allocatd on CPU
                            return self.runtime_client.create_host_memref_view(
                                int(arr.ctypes.data),
                                shape=list(self.shape),
                                dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            )
                        else:
                            assert hasattr(arr, "__cuda_array_interface__")
                            # Jax tensor allocatd on GPU
                            memref_value = self.runtime_client.create_device_memref_view(
                                int(arr.data.ptr),
                                shape=list(self.shape),
                                dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                                device=self.runtime_client.get_devices()[0],
                            )
                            return self.runtime_client.copy_to_host(
                                host_memref=memref_value,
                            )
                    else:
                        assert 0 and "Conversion to memref is not supported."
                else:
                    assert self.device == tp_device("gpu")
                    if hasattr(data, "__array_interface__"):
                        memref_value = self.runtime_client.create_host_memref_view(
                            int(data.ctypes.data),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                        )
                        return self.runtime_client.copy_to_device(
                            host_memref=memref_value,
                            device=self.runtime_client.get_devices()[0],
                        )
                    elif hasattr(data, "data_ptr") and data.device.type == "cuda":
                        return self.runtime_client.create_device_memref_view(
                            int(data.data_ptr()),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            device=self.runtime_client.get_devices()[0],
                        )
                    elif hasattr(data, "data_ptr") and data.device.type == "cpu":
                        memref_value = self.runtime_client.create_host_memref_view(
                            int(data.data_ptr()),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                        )
                        return self.runtime_client.copy_to_device(
                            host_memref=memref_value,
                            device=self.runtime_client.get_devices()[0],
                        )
                    elif hasattr(data, "data") and hasattr(data, "__cuda_array_interface__"):
                        return self.runtime_client.create_device_memref_view(
                            int(data.data.ptr),
                            shape=list(self.shape),
                            dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            device=self.runtime_client.get_devices()[0],
                        )
                    elif hasattr(data, "__array__"):
                        arr = data.__array__()
                        if hasattr(arr, "__cuda_array_interface__"):
                            arr
                            # Jax tensor allocatd on GPU
                            return self.runtime_client.create_device_memref_view(
                                int(arr.data.ptr),
                                shape=list(self.shape),
                                dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                                device=self.runtime_client.get_devices()[0],
                            )
                        else:
                            assert hasattr(arr, "__array_interface__")
                            # Jax tensor allocatd on CPU
                            memref_value = self.runtime_client.create_host_memref_view(
                                int(arr.ctypes.data),
                                shape=list(self.shape),
                                dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                            )
                            return self.runtime_client.copy_to_device(
                                host_memref=memref_value,
                                device=self.runtime_client.get_devices()[0],
                            )
                    else:
                        assert 0 and "Conversion to memref is not supported."

    def __dlpack__(self, stream=None):
        return self.memref_value.__dlpack__()

    def __dlpack_device__(self):
        return self.memref_value.__dlpack_device__()

    def __str__(self):
        return self._prettyprint()

    def __eq__(self, other) -> bool:
        """
        Check if two arrays are equal.

        Args:
            other (Array): Another Array object for comparison.

        Returns:
            bool: True if the arrays are equal, False otherwise.
        """
        return self.data() == other.data()
