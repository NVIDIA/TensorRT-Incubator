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
import torch

import tripy.common.datatype

from tests import helper
from tripy.common.exception import TripyException
from tripy.common.utils import (
    convert_list_to_array,
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
def convert_list_to_array(values, dtype, expected):
    result = convert_list_to_array(values, dtype)
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
