from typing import Any, List, Optional, Tuple, Union

import cupy as cp
import numpy as np

from tripy import utils
from tripy.common.device import device as tp_device
from tripy.common.exception import raise_error


def convert_tripy_to_module_dtype(dtype: "tripy.common.datatype.dtype", module) -> Any:
    """
    Get the numpy equivalent of tripy.common.datatype.
    """
    import tripy.common.datatype

    TRIPY_TO_NUMPY = dict(
        {
            tripy.common.datatype.float32: module.float32,
            tripy.common.datatype.int32: module.int32,
            tripy.common.datatype.int8: module.int8,
            tripy.common.datatype.int64: module.int64,
            tripy.common.datatype.uint8: module.uint8,
            tripy.common.datatype.float16: module.float16,
            tripy.common.datatype.bool: module.bool_,
            tripy.common.datatype.float8: module.uint8,
            tripy.common.datatype.int4: module.uint8,
        }
    )

    return TRIPY_TO_NUMPY[dtype]


def convert_to_tripy_dtype(dtype: Any) -> Optional["tripy.common.datatype.dtype"]:
    """
    Get the tripy.common.datatype equivalent of the data type.
    """
    import tripy.common.datatype

    PYTHON_NATIVE_MAPPING = {int: tripy.common.datatype.int32, float: tripy.common.datatype.float32}
    if dtype in PYTHON_NATIVE_MAPPING:
        return PYTHON_NATIVE_MAPPING[dtype]

    try:
        dtype_name = dtype.name
    except AttributeError:
        dtype_name = str(dtype).split(".", 1)[-1].strip("'>")

    NUMPY_TO_TRIPY = {
        "int8": tripy.common.datatype.int8,
        "int32": tripy.common.datatype.int32,
        "int64": tripy.common.datatype.int64,
        "uint8": tripy.common.datatype.uint8,
        "float16": tripy.common.datatype.float16,
        "float32": tripy.common.datatype.float32,
        "bool": tripy.common.datatype.bool,
    }

    return NUMPY_TO_TRIPY.get(dtype_name, None)


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
        data: Union[List, np.ndarray, cp.ndarray, "torch.Tensor", "jnp.ndarray"],
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
        self._module = np if self.device.kind == "cpu" else cp

        if data is None:
            if shape is None:
                raise_error("Shape must be provided when data is None.", [])
            # Allocate dummy data
            data = self._module.empty(
                dtype=convert_tripy_to_module_dtype(
                    utils.default(dtype, tripy.common.datatype.float32), module=self._module
                ),
                shape=shape,
            )
        else:
            if isinstance(data, (List, int, float, tuple)):

                def get_element_type(elements):
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
                            details=[
                                f"List element type can only be int or float.",
                                f"Got element {e} of type {type(e)}.",
                            ],
                        )

                element_type = get_element_type(data)
                # allow casting for python types
                data = self._module.array(
                    data,
                    dtype=convert_tripy_to_module_dtype(dtype, self._module) if dtype is not None else element_type,
                )

            data_dtype = convert_to_tripy_dtype(data.dtype)
            if not data_dtype:
                raise_error(f"Data has unsupported dtype: {data.dtype}")

            # Check for consistency if dtype/shape was provided with data:
            # TODO(#161): Remove the exception for fp8 and int4
            if (
                dtype
                not in (
                    None,
                    tripy.common.datatype.float8,
                    tripy.common.datatype.int4,
                )
                and data_dtype != dtype
            ):
                raise_error(
                    "Data has incorrect dtype.",
                    details=[f"Input data had type: {data_dtype}, ", f"but provided dtype was: {dtype}"],
                )

            if shape is not None and shape != data.shape:
                raise_error(
                    "Data has incorrect shape.",
                    details=[
                        f"Input data had shape: {data.shape}, ",
                        f"but provided runtime shape was: {shape}",
                    ],
                )

        self.shape = data.shape
        self.dtype = dtype if dtype is not None else convert_to_tripy_dtype(data.dtype)

        # Convert data with correct dtype and shape to a byte buffer.
        def convert_to_byte_buffer(data):
            if not data.shape:
                # Numpy requires reshaping to 1d because of the following error:
                #    "ValueError: Changing the dtype of a 0d array is only supported if the itemsize is unchanged"
                data = data.reshape((1,))
            return self._module.array(data).view(self._module.uint8)

        self.byte_buffer: Union[np.ndarray, cp.ndarray] = convert_to_byte_buffer(data)

    def view(self):
        """
        Create a NumPy Or CuPy array of underlying datatype.
        """
        assert self.dtype is not None
        from tripy.common.datatype import DATA_TYPES

        assert self.dtype.name in DATA_TYPES
        dtype = convert_tripy_to_module_dtype(self.dtype, module=self._module)
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
