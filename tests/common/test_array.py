
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

from typing import Any, List

import cupy as cp
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from textwrap import dedent

import mlir_tensorrt.runtime.api as runtime
import tripy as tp

from tests.helper import NUMPY_TYPES, torch_type_supported
from tripy.common.array import Array
from tripy.common.datatype import DATA_TYPES
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype, get_supported_type_for_python_sequence
from tripy.backend.mlir.utils import convert_tripy_dtype_to_runtime_dtype

data_list = []

# Create a data list for NumPy arrays
np_data = [np.ones(1, dtype=dtype) for dtype in NUMPY_TYPES]
# also test empty instantiation
np_data.extend(np.ones(0, dtype=dtype) for dtype in NUMPY_TYPES)
data_list.extend(np_data)

# Extend the data list for Cupy arrays
data_list.extend([cp.array(data) for data in np_data])

# Extend the data list for Torch CPU tensors
data_list.extend([torch.tensor(data) for data in filter(torch_type_supported, np_data)])

# Extend the data list for Torch GPU tensors
data_list.extend([torch.tensor(data).to(torch.device("cuda")) for data in filter(torch_type_supported, np_data)])

# Extend the data list for Jax CPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cpu")[0]) for data in np_data])

# Extend the data list for Jax GPU arrays
data_list.extend([jax.device_put(jnp.array(data), jax.devices("cuda")[0]) for data in np_data])


class TestArray:
    @pytest.mark.parametrize("input_data", data_list, ids=lambda data: f"{type(data).__qualname__}")
    def test_creation(self, input_data):
        """
        Test the creation of Array objects with different devices and data types.
        """
        dtype = convert_frontend_dtype_to_tripy_dtype(input_data.dtype)
        shape = input_data.shape
        if dtype is not None:
            arr = Array(input_data, shape, dtype)
            assert isinstance(arr, Array)
            assert isinstance(arr.memref_value, runtime.MemRefValue)
            assert arr.memref_value.dtype == convert_tripy_dtype_to_runtime_dtype(dtype)
            assert (
                arr.memref_value.address_space == runtime.PointerType.host
                if arr.device.kind == "cpu"
                else runtime.PointerType.device
            )

    @pytest.mark.parametrize("np_dtype", NUMPY_TYPES)
    def test_0d(self, np_dtype):
        dtype = convert_frontend_dtype_to_tripy_dtype(np_dtype)
        arr = Array(np.array(1, dtype=np_dtype), None, dtype, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == tuple()
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_nested_list(self, dtype):
        arr = Array([[1, 2], [3, 4]], None, dtype, tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (2, 2)
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_empty_dimension(self, dtype):
        arr = Array([[], [], []], shape=(3, 0), dtype=dtype, device=tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (
            3,
            0,
        )
        assert arr.dtype == dtype

    @pytest.mark.parametrize("dtype", [tp.float32, tp.int32, tp.int64, tp.bool])
    def test_later_empty_dimensions(self, dtype):
        # consistent to call it (3, 0, ...) because it can't be checked further
        arr = Array([[], [], []], shape=(3, 0, 1, 2), dtype=dtype, device=tp.device("cpu"))
        assert isinstance(arr, Array)
        assert arr.shape == (3, 0, 1, 2)
        assert arr.dtype == dtype

    def test_missing_data_shape(self):
        with pytest.raises(tp.TripyException, match="Shape must be provided when data is None.") as exc:
            _ = Array(None, None, tp.float32, tp.device("cpu"))
        print(str(exc.value))

    def test_missing_data_dtype(self):
        with pytest.raises(tp.TripyException, match="Datatype must be provided when data is None.") as exc:
            _ = Array(None, None, None, tp.device("cpu"))
        print(str(exc.value))

    def test_missing_dtype_for_empty_array(self):
        with pytest.raises(tp.TripyException, match="Datatype must be provided for empty data") as exc:
            _ = Array([[[], []], [[], []]])
        print(str(exc.value))

    def test_incorrect_dtype(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect dtype.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), None, tp.float32, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array(np.ones((2,), dtype=np.int32), (3,), None, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape_list(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array((1, 2, 3), (5,), None, tp.device("cpu"))
        print(str(exc.value))

    def test_incorrect_shape_nested_list(self):
        with pytest.raises(tp.TripyException, match="Data has incorrect shape.") as exc:
            _ = Array([[1, 2, 3], [4, 5, 6]], (1, 2, 3), None, tp.device("cpu"))
        print(str(exc.value))

    def test_inconsistent_jagged_list(self):
        with pytest.raises(tp.TripyException, match="Length of list at index 1 does not match at dimension 1") as exc:
            _ = Array([[1, 2, 3], [4, 5]], (2, 3), None, tp.device("cpu"))
        print(str(exc.value))

    def test_inconsistent_scalar_value(self):
        with pytest.raises(tp.TripyException, match="Expected sequence at index 1, got int") as exc:
            _ = Array([[1, 2, 3], 0], (2, 3), None, tp.device("cpu"))
        print(str(exc.value))

    def test_unsupported_list_element(self):
        from decimal import Decimal

        with pytest.raises(tp.TripyException, match="List element type can only be int or float.") as exc:
            _ = Array([Decimal(0)], None, None, tp.device("cpu"))
        print(str(exc.value))

    @pytest.mark.parametrize("dtype", get_supported_type_for_python_sequence())
    def test_array_supported_python_sequence_type(self, dtype):
        arr = Array([0], shape=None, dtype=dtype, device=tp.device("cpu"))
        assert isinstance(arr.memref_value, runtime.MemRefValue)
        assert arr.memref_value.dtype == convert_tripy_dtype_to_runtime_dtype(dtype)
        assert arr.memref_value.address_space == runtime.PointerType.host

    @pytest.mark.parametrize(
        "dtype",
        [dtype for dtype in DATA_TYPES.values() if dtype not in get_supported_type_for_python_sequence()]
        + ["unsupported_type"],
    )
    def test_array_unsupported_python_sequence_type(self, dtype):
        with pytest.raises(AssertionError):
            arr = Array([0], shape=None, dtype=dtype, device=tp.device("cpu"))
