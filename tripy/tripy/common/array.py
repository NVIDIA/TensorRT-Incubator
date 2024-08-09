#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Any, List, Optional, Sequence, Tuple, Union
import array

from tripy import utils
from tripy.common.device import device as tp_device
from tripy.common.exception import raise_error
from tripy.common.utils import (
    convert_frontend_dtype_to_tripy_dtype,
    Float16MemoryView,
)
import mlir_tensorrt.runtime.api as runtime
import tripy.common.datatype


def has_no_contents(data: Any) -> bool:
    while isinstance(data, Sequence):
        if len(data) == 0:
            return True
        data = data[0]
    return False


def check_dtype_consistency(actual_dtype, stated_dtype) -> None:
    if stated_dtype is None:
        return

    convert_dtype = convert_frontend_dtype_to_tripy_dtype(actual_dtype)
    if not convert_dtype:
        raise_error(f"Data has unsupported dtype: {actual_dtype}")
    if convert_dtype != stated_dtype:
        raise_error(
            "Data has incorrect dtype.",
            details=[f"Input data had type: {convert_dtype}, ", f"but provided dtype was: {stated_dtype}"],
        )


def check_shape_consistency(actual_shape, stated_shape):
    if stated_shape is None:
        return

    # cut off after a 0 dimension because there would be no further elements to compare
    # so any values would be consistent with each other
    def equal_up_to_0(actual_shape, stated_shape):
        for i in range(max(len(actual_shape), len(stated_shape))):
            # lengths differ but not because we hit a 0
            if i >= len(actual_shape) or i >= len(stated_shape):
                return False
            if actual_shape[i] != stated_shape[i]:
                return False
            if actual_shape[i] == 0:
                break
        return True

    if not equal_up_to_0(actual_shape, stated_shape):
        raise_error(
            "Data has incorrect shape.",
            details=[
                f"Input data had shape: {actual_shape}, ",
                f"but provided runtime shape was: {stated_shape}",
            ],
        )


def check_data_consistency(actual_shape, actual_dtype, stated_shape, stated_dtype):
    check_dtype_consistency(actual_dtype, stated_dtype)
    check_shape_consistency(actual_shape, stated_shape)


def check_list_consistency(input_list: Any, computed_shape: List[int]) -> None:
    def recursive_helper(input_list, dim_idx, *indices):
        # compute only if we need it
        def describe_indices():
            if len(indices) == 0:
                # this should not happen if we are computing the shape from the list
                return "the top level"
            if len(indices) == 1:
                return f"index {str(indices[0])}"
            return f"indices [{', '.join(map(str, indices))}]"

        # base case: last dimension iff scalar
        if not isinstance(input_list, Sequence):
            return (
                dim_idx >= len(computed_shape),
                f"Expected sequence at {describe_indices()}, got {type(input_list).__name__}",
            )
        if dim_idx >= len(computed_shape):
            return (
                not isinstance(input_list, Sequence),
                f"Expected scalar at {describe_indices()}, got {type(input_list).__name__}",
            )

        if len(input_list) != computed_shape[dim_idx]:
            return (False, f"Length of list at {describe_indices()} does not match at dimension {dim_idx}")

        # we should not encounter anything past the 0 if the shape was computed from the list
        if computed_shape[dim_idx] == 0:
            return True, ""

        for i, child in enumerate(input_list):
            check_result, msg = recursive_helper(child, dim_idx + 1, *indices, i)
            if not check_result:
                return (False, msg)
        return True, ""

    res, msg = recursive_helper(input_list, 0)
    if not res:
        raise_error(msg, details=[f"Input list: {input_list}\n", f"Expected shape: {computed_shape}"])


# The class abstracts away implementation differences between Torch, Jax, Cupy, NumPy, and List.
# Data is stored as a byte buffer, enabling interoperability across array libraries.
# The byte buffer is created using the `convert_to_byte_buffer` function.
# Views with different data types can be created using the `view` method.
class Array:
    """
    A versatile array container that works with Torch, Jax, Cupy, NumPy.
    It can be used to store any object implementing dlpack interface.
    """

    def __init__(
        self,
        data: Union["np.ndarray", "cp.ndarray", "torch.Tensor", "jnp.ndarray"],
        shape: Optional[Tuple[int]] = None,
        dtype: "tripy.dtype" = None,
        device: tp_device = None,
    ) -> None:
        """
        Initialize an Array object.

        Args:
            data: Input data that implements dlpack interface such as np.ndarray, cp.ndarray, torch.Tensor, or jnp.ndarray.
            dtype: Data type of the array.
            shape: Shape information for static allocation.
            device: Target device (tripy.Device("cpu") or tripy.Device("gpu")).
        """
        import tripy.common.datatype
        from tripy.backend.mlir.utils import MLIRRuntimeClient

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
            check_data_consistency(data.shape, data.dtype, shape, dtype)
            self.dtype = convert_frontend_dtype_to_tripy_dtype(data.dtype)
            self.shape = data.shape

        # Store the memref_value
        self.runtime_client = MLIRRuntimeClient()
        self.data_ref = data  # Ensure that data does not go out of scope when we create a view over it.
        self.memref_value = self._memref(data)
        self.device = (
            tp_device("gpu") if self.memref_value.address_space == runtime.PointerType.device else tp_device("cpu")
        )

    def data(self) -> List[Union[float, int]]:
        memref = self.memref_value
        if self.memref_value.address_space == runtime.PointerType.device:
            memref = self.runtime_client.copy_to_host(
                device_memref=self.memref_value,
            )
        try:
            return memoryview(memref).tolist()
        except NotImplementedError as e:
            if "memoryview: format e not supported" in str(e):
                assert memref.dtype == runtime.ScalarTypeCode.f16
                return Float16MemoryView(bytearray(memref)).tolist()
            raise

    def _prettyprint(self, threshold=1000, linewidth=10, edgeitems=3):
        data = self.data()

        numel = 1
        for d in self.shape:
            numel *= d
        summarize = numel > threshold
        return self._data_str(data, summarize, linewidth, edgeitems)

    def _data_str(self, data, summarize, linewidth, edgeitems, indent=0):
        if isinstance(data, (float, int)):
            return str(data)

        if len(data) == 0 or isinstance(data[0], (float, int)):
            if summarize and len(data) > 2 * edgeitems:
                data_lines = [data[:edgeitems] + [" ..."] + data[-edgeitems:]]
            else:
                data_lines = [data[i : i + linewidth] for i in range(0, len(data), linewidth)]
            lines = [", ".join([f"{e:.4f}" if isinstance(e, float) else str(e) for e in line]) for line in data_lines]
            return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"

        if summarize and len(data) > 2 * edgeitems:
            slices = (
                [self._data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, edgeitems)]
                + ["..."]
                + [
                    self._data_str(data[i], summarize, linewidth, edgeitems, indent + 1)
                    for i in range(len(data) - edgeitems, len(data))
                ]
            )
        else:
            slices = [self._data_str(data[i], summarize, linewidth, edgeitems, indent + 1) for i in range(0, len(data))]

        tensor_str = ("," + "\n" * (max(len(self.shape) - indent - 1, 1)) + " " * (indent + 1)).join(slices)
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
                raise_error(f"Cannot allocate tensor that is currently on: {device} on requested device: {self.device}")

            # a pointer value of 0 is used only for empty tensors
            if ptr == 0:
                assert 0 in list(
                    self.shape
                ), f"Recieved null pointer for buffer but tensor is not empty (shape {list(self.shape)})"
                self.device = utils.default(self.device, tp_device("gpu"))
                mlirtrt_device = self.runtime_client.get_devices()[0] if self.device == tp_device("gpu") else None

                return self.runtime_client.create_memref(
                    array.array("i", []),  # typecode is not important because it's empty
                    shape=list(self.shape),
                    dtype=convert_tripy_dtype_to_runtime_dtype(self.dtype),
                    device=mlirtrt_device,
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

    def __repr__(self):
        return str(self)

    def __eq__(self, other) -> bool:
        """
        Check if two arrays are equal.

        Args:
            other (Array): Another Array object for comparison.

        Returns:
            bool: True if the arrays are equal, False otherwise.
        """
        return self.data() == other.data()
