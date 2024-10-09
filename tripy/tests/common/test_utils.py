#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
        ([100000, 200000], tripy.common.datatype.int32, b"\xa0\x86\x01\x00@\x0d\x03\x00"),
        ([1, 2], tripy.common.datatype.int64, b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"),
        ([1.0, 2.0], tripy.common.datatype.float32, b"\x00\x00\x80?\x00\x00\x00@"),
    ],
)
def test_convert_list_to_array(values, dtype, expected):
    result = convert_list_to_array(values, dtype)
    assert result.tobytes() == expected
