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

import pytest
import struct
from collections import ChainMap
from textwrap import dedent

import cupy as cp
import numpy as np
import jax.numpy as jnp
import torch

import tripy.common.datatype

from tests import helper

from tripy.common.exception import TripyException
from tripy.common.utils import (
    convert_frontend_dtype_to_tripy_dtype,
    convert_list_to_bytebuffer,
    Float16MemoryView,
    get_element_type,
)


def test_get_element_type():
    assert get_element_type([1, 2, 3]) == tripy.common.datatype.int32
    assert get_element_type([1.0, 2.0, 3.0]) == tripy.common.datatype.float32
    assert get_element_type([[1], [2], [3]]) == tripy.common.datatype.int32

    with helper.raises(
        TripyException,
        match="Unsupported element type.",
    ):
        get_element_type(["a", "b", "c"])


def test_convert_frontend_dtype_to_tripy_dtype():
    import numpy as np

    assert convert_frontend_dtype_to_tripy_dtype(tripy.common.datatype.int32) == tripy.common.datatype.int32

    PYTHON_NATIVE_TO_TRIPY = {
        int: tripy.common.datatype.int32,
        float: tripy.common.datatype.float32,
        bool: tripy.common.datatype.bool,
    }

    CUPY_TO_TRIPY = {
        cp.bool_: tripy.common.datatype.bool,
        cp.int8: tripy.common.datatype.int8,
        cp.int32: tripy.common.datatype.int32,
        cp.int64: tripy.common.datatype.int64,
        cp.float16: tripy.common.datatype.float16,
        cp.float32: tripy.common.datatype.float32,
    }

    NUMPY_TO_TRIPY = {
        np.bool_: tripy.common.datatype.bool,
        np.int8: tripy.common.datatype.int8,
        np.int32: tripy.common.datatype.int32,
        np.int64: tripy.common.datatype.int64,
        np.float16: tripy.common.datatype.float16,
        np.float32: tripy.common.datatype.float32,
    }

    TORCH_TO_TRIPY = {
        torch.bool: tripy.common.datatype.bool,
        torch.int8: tripy.common.datatype.int8,
        torch.int32: tripy.common.datatype.int32,
        torch.int64: tripy.common.datatype.int64,
        torch.float16: tripy.common.datatype.float16,
        torch.bfloat16: tripy.common.datatype.bfloat16,
        torch.float32: tripy.common.datatype.float32,
    }

    JAX_TO_TRIPY = {
        jnp.bool_: tripy.common.datatype.bool,
        jnp.int8: tripy.common.datatype.int8,
        jnp.int32: tripy.common.datatype.int32,
        jnp.int64: tripy.common.datatype.int64,
        jnp.float8_e4m3fn: tripy.common.datatype.float8,
        jnp.float16: tripy.common.datatype.float16,
        jnp.bfloat16: tripy.common.datatype.bfloat16,
        jnp.float32: tripy.common.datatype.float32,
    }

    FRONTEND_TO_TRIPY = dict(ChainMap(PYTHON_NATIVE_TO_TRIPY, NUMPY_TO_TRIPY, TORCH_TO_TRIPY, JAX_TO_TRIPY))

    for frontend_type, tripy_type in FRONTEND_TO_TRIPY.items():
        assert convert_frontend_dtype_to_tripy_dtype(frontend_type) == tripy_type

    for unsupported_type in [
        "unsupported_type",
        np.int16,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float64,
        cp.int16,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float64,
        torch.int16,
        torch.float64,
        jnp.int4,
        jnp.int16,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.float64,
    ]:
        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Unsupported data type: '{unsupported_type}'.
                TriPy tensors can be constructed with one of the following data types: int, float, bool, bool_, int8, int32, int64, float8_e4m3fn, float16, bfloat16, float32.
            """
            ).strip(),
        ):
            convert_frontend_dtype_to_tripy_dtype(unsupported_type)


@pytest.mark.parametrize(
    "values, dtype, expected",
    [
        ([True, False, True], tripy.common.datatype.bool, b"\x01\x00\x01"),
        ([1, 2, 3], tripy.common.datatype.int8, b"\x01\x02\x03"),
        ([100000, 200000], tripy.common.datatype.int32, b"\xa0\x86\x01\x00@\x0d\x03\x00"),
        ([1, 2], tripy.common.datatype.int64, b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"),
        ([1.0, 2.0], tripy.common.datatype.float16, b"\x00<\x00@"),
        ([1.0, 2.0], tripy.common.datatype.float32, b"\x00\x00\x80?\x00\x00\x00@"),
    ],
)
def test_convert_list_to_bytebuffer(values, dtype, expected):
    result = convert_list_to_bytebuffer(values, dtype)
    assert result == expected


def test_float16_memoryview():
    memview = Float16MemoryView(bytearray(struct.pack("5e", 1.5, 2.5, 3.5, 4.5, 5.5)))
    assert memview.itemsize == 2
    assert memview.format == "e"
    len(memview) == 5
    assert memview[0] == pytest.approx(1.5)
    assert memview[2] == pytest.approx(3.5)
    assert memview[1:4] == pytest.approx([2.5, 3.5, 4.5])
    expected = [1.5, 2.5, 3.5, 4.5, 5.5]
    assert memview.tolist() == pytest.approx(expected)

    # Largest representable value in float16
    large_value = 65504.0
    buffer = struct.pack("e", large_value)
    mv = Float16MemoryView(buffer)
    assert mv[0] == pytest.approx(large_value)

    # Smallest positive normal number for float16
    small_value = 6.1035e-5
    buffer = struct.pack("e", small_value)
    mv = Float16MemoryView(buffer)
    assert mv[0] == pytest.approx(small_value, rel=1e-3)

    # Negative value
    negative_value = -42.5
    buffer = struct.pack("e", negative_value)
    mv = Float16MemoryView(buffer)
    assert mv[0] == pytest.approx(negative_value)
