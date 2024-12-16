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

import array
from typing import Any, List, Sequence

import nvtripy.common.datatype
from nvtripy.common.exception import raise_error


def is_int32(data):
    return nvtripy.common.datatype.INT32_MIN <= data <= nvtripy.common.datatype.INT32_MAX


def get_element_type(elements):
    e = elements
    while (isinstance(e, List) or isinstance(e, tuple)) and len(e) > 0:
        e = e[0]
    if isinstance(e, bool):
        dtype = nvtripy.common.datatype.bool
    elif isinstance(e, int):
        if is_int32(e):
            dtype = nvtripy.common.datatype.int32
        else:
            dtype = nvtripy.common.datatype.int64
    elif isinstance(e, float):
        dtype = nvtripy.common.datatype.float32
    # Special handling for empty tensors
    elif isinstance(e, list) or isinstance(e, tuple):
        dtype = None
    else:
        raise_error(
            "Unsupported element type.",
            details=[
                f"List element type can only be int or float. ",
                f"Got element {e} of type {type(e)}.",
            ],
        )

    return dtype


def convert_list_to_array(values: List[Any], dtype: str) -> bytes:
    """Convert a list of values to a byte buffer based on the specified dtype."""
    # Lookup table for types and their corresponding struct format characters
    TYPE_TO_FORMAT = {
        nvtripy.common.datatype.bool: "b",
        nvtripy.common.datatype.int64: "l",
        nvtripy.common.datatype.int32: "i",
        nvtripy.common.datatype.float32: "f",
    }
    if dtype not in TYPE_TO_FORMAT:
        raise ValueError(f"Unsupported type: {dtype}")

    return array.array(TYPE_TO_FORMAT[dtype], values)


def is_empty(data: Sequence) -> bool:
    return isinstance(data, Sequence) and all(map(is_empty, data))
