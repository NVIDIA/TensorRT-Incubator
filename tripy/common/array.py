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
        dtype: "tripy.dtype" = None,
        shape: Optional[Tuple[int]] = None,
        device: tp_device = None,
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

        self.device = device

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
        self.device = (
            tp_device("gpu") if self.memref_value.address_space == runtime.PointerType.device else tp_device("cpu")
        )

    def data(self) -> List[Union[float, int]]:
        if self.memref_value.address_space == runtime.PointerType.device:
            host_memref = self.runtime_client.copy_to_host(
                device_memref=self.memref_value,
            )
            return memoryview(host_memref).tolist()
        return memoryview(self.memref_value).tolist()

    def _prettyprint(self, threshold=1000, linewidth=10, edgeitems=3):
        data = self.data()

        numel = 1
        for d in self.shape:
            numel *= d
        summarize = numel > threshold

        return self._data_str(data, 0, summarize, linewidth, edgeitems)

    def _data_str(self, data, indent, summarize, linewidth, edgeitems):
        if isinstance(data, (float, int)):
            return str(data)

        if len(data) == 0 or isinstance(data[0], (float, int)):
            if summarize and len(data) > 2 * edgeitems:
                data_lines = [data[:edgeitems] + [" ..."] + data[-edgeitems:]]
            else:
                data_lines = [data[i : i + linewidth] for i in range(0, len(data), linewidth)]
            lines = [", ".join([str(e) for e in line]) for line in data_lines]
            return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

        if summarize and len(data) > 2 * edgeitems:
            slices = (
                [self._data_str(data[i], indent + 1, summarize, linewidth, edgeitems) for i in range(0, edgeitems)]
                + ["..."]
                + [
                    self._data_str(data[i], indent + 1, summarize, linewidth, edgeitems)
                    for i in range(len(data) - edgeitems, len(data))
                ]
            )
        else:
            slices = [self._data_str(data[i], indent + 1, summarize, linewidth, edgeitems) for i in range(0, len(data))]

        tensor_str = ("," + "\n" * (len(self.shape) - 1) + " " * (indent + 1)).join(slices)
        return "[" + tensor_str + "]"

    def _memref(self, data):
        from tripy.backend.mlir.utils import convert_tripy_dtype_to_runtime_dtype

        if data is None:
            self.device = utils.default(self.device, tp_device("gpu"))
            mlirtrt_device = self.runtime_client.get_devices()[0] if self.device == tp_device("gpu") else None

            return self.runtime_client.create_memref(
                shape=list(self.shape), dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype), device=mlirtrt_device
            )
        else:

            if isinstance(data, (List, tuple, int, float)):
                self.device = utils.default(self.device, tp_device("gpu"))
                mlirtrt_device = (
                    self.runtime_client.get_devices()[self.device.index] if self.device == tp_device("gpu") else None
                )

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

                def can_access_attr(attr_name):
                    try:
                        getattr(data, attr_name)
                    except:
                        return False
                    return True

                ptr = None
                if hasattr(data, "__array_interface__"):
                    ptr = int(data.ctypes.data)
                    device = tp_device("cpu")
                elif hasattr(data, "data_ptr"):
                    ptr = int(data.data_ptr())
                    device = tp_device(
                        ("cpu" if data.device.type == "cpu" else "gpu")
                        + (":" + str(data.device.index) if data.device.index is not None else "")
                    )
                elif can_access_attr("__cuda_array_interface__"):
                    ptr = int(data.__cuda_array_interface__["data"][0])
                    device = tp_device("gpu")
                elif hasattr(data, "__array__"):
                    ptr = data.__array__().ctypes.data
                    device = tp_device("cpu")
                else:
                    raise_error(f"Unsupported type: {data}")

                self.device = utils.default(self.device, device)
                if self.device != device:
                    raise_error(
                        f"Cannot allocate tensor that is currently on: {device} on requested device: {self.device}"
                    )

                if self.device.kind == "cpu":
                    return self.runtime_client.create_host_memref_view(
                        ptr,
                        shape=list(self.shape),
                        dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                    )
                else:
                    return self.runtime_client.create_device_memref_view(
                        ptr,
                        shape=list(self.shape),
                        dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                        device=self.runtime_client.get_devices()[self.device.index],
                    )

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
