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

import re
import struct
from typing import Any, List, Optional, Tuple, Union

from tripy.common.exception import raise_error
from tripy.logging import logger
from tripy.common.datatype import float32, int32, int64
from tripy.common.datatype import bool as tp_bool
import tripy.common.datatype
from tripy.common.datatype import DATA_TYPES


def get_supported_type_for_python_sequence() -> List["tripy.common.datatype"]:
    return [
        t
        for t in DATA_TYPES.values()
        if t not in [tripy.common.datatype.int4, tripy.common.datatype.float8, tripy.common.datatype.bfloat16]
    ]


def is_int32(data):
    return tripy.common.datatype.INT32_MIN <= data <= tripy.common.datatype.INT32_MAX


def get_element_type(elements):
    e = elements
    while (isinstance(e, List) or isinstance(e, tuple)) and len(e) > 0:
        e = e[0]
    if isinstance(e, bool):
        return tripy.common.datatype.bool
    if isinstance(e, int):
        if is_int32(e):
            return tripy.common.datatype.int32
        return tripy.common.datatype.int64
    elif isinstance(e, float):
        return tripy.common.datatype.float32
    # Special handling for empty tensors
    elif isinstance(e, list) or isinstance(e, tuple):
        return None
    else:
        raise_error(
            "Unsupported element type.",
            details=[
                f"List element type can only be int or float. ",
                f"Got element {e} of type {type(e)}.",
            ],
        )


def convert_frontend_dtype_to_tripy_dtype(dtype: Any) -> Optional["tripy.common.datatype.dtype"]:
    """
    Get the tripy.common.datatype equivalent of the data type.
    """
    import tripy.common.datatype

    if isinstance(dtype, tripy.common.datatype.dtype):
        return dtype

    try:
        dtype_name = dtype.name
    except AttributeError:

        def _convert_class_string(class_string):
            pattern = r"<class '([\w.]+)'>$"
            match = re.match(pattern, class_string)
            return match.group(1) if match else class_string

        def _extract_type_name(type_string):
            pattern = r"(?:.*\.)?(\w+)$"
            match = re.match(pattern, type_string)
            return match.group(1) if match else type_string

        dtype_name = _extract_type_name(_convert_class_string(str(dtype)))

    DTYPE_NAME_TO_TRIPY = {
        # Native python types.
        "int": tripy.common.datatype.int32,
        "float": tripy.common.datatype.float32,
        "bool": tripy.common.datatype.bool,
        # Framework types.
        "bool_": tripy.common.datatype.bool,
        "int8": tripy.common.datatype.int8,
        "int32": tripy.common.datatype.int32,
        "int64": tripy.common.datatype.int64,
        "float8_e4m3fn": tripy.common.datatype.float8,
        "float16": tripy.common.datatype.float16,
        "bfloat16": tripy.common.datatype.bfloat16,
        "float32": tripy.common.datatype.float32,
    }

    converted_type = DTYPE_NAME_TO_TRIPY.get(dtype_name, None)
    if not converted_type:
        raise_error(
            f"Unsupported data type: '{dtype}'.",
            [
                f"Tripy tensors can be constructed with one of the following data types: {', '.join(DTYPE_NAME_TO_TRIPY.keys())}."
            ],
        )

    return converted_type


def convert_list_to_bytebuffer(values: List[Any], dtype: str) -> bytes:
    """Convert a list of values to a byte buffer based on the specified dtype."""

    # Lookup table for types and their corresponding struct format characters
    TYPE_TO_FORMAT = {
        tripy.common.datatype.bool: "?",
        tripy.common.datatype.int8: "b",
        tripy.common.datatype.int32: "i",
        tripy.common.datatype.int64: "q",
        tripy.common.datatype.float16: "e",
        tripy.common.datatype.float32: "f",
    }

    if dtype not in TYPE_TO_FORMAT:
        raise ValueError(f"Unsupported type: {dtype}")

    buffer = bytearray()

    for value in values:
        format_char = TYPE_TO_FORMAT[dtype]
        buffer.extend(struct.pack(f"<{format_char}", value))

    return bytes(buffer)
