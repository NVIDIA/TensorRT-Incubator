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

import array
import struct
from typing import Any, List, Sequence

from tripy.common.exception import raise_error
import tripy.common.datatype


def is_int32(data):
    return tripy.common.datatype.INT32_MIN <= data <= tripy.common.datatype.INT32_MAX


def get_element_type(elements):
    e = elements
    while (isinstance(e, List) or isinstance(e, tuple)) and len(e) > 0:
        e = e[0]
    if isinstance(e, bool):
        dtype = tripy.common.datatype.bool
    elif isinstance(e, int):
        if is_int32(e):
            dtype = tripy.common.datatype.int32
        else:
            dtype = tripy.common.datatype.int64
    elif isinstance(e, float):
        dtype = tripy.common.datatype.float32
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
        tripy.common.datatype.bool: "b",
        tripy.common.datatype.int64: "l",
        tripy.common.datatype.int32: "i",
        tripy.common.datatype.float32: "f",
    }
    if dtype not in TYPE_TO_FORMAT:
        raise ValueError(f"Unsupported type: {dtype}")

    return array.array(TYPE_TO_FORMAT[dtype], values)


def has_no_contents(data: Sequence) -> bool:
    while isinstance(data, Sequence):
        if len(data) == 0:
            return True
        data = data[0]
    return False


class Float16MemoryView:
    """
    A custom memory view class for handling float16 data.
    """

    def __init__(self, buffer):
        """
        Initialize the Float16MemoryView with a buffer.

        Args:
            buffer (buffer): The buffer containing float16 data.
        """
        self.buffer = buffer
        self.itemsize = 2  # size of float16 in bytes
        self.format = "e"  # format character for float16

    def __getitem__(self, index):
        """
        Get an item or a slice from the buffer.

        Args:
            index (int or slice): The index or slice to retrieve.

        Returns:
            float or list of floats: The float16 value(s) at the specified index or slice.
        """
        if isinstance(index, slice):
            return [
                self._unpack(self.buffer[i * self.itemsize : i * self.itemsize + self.itemsize])
                for i in range(*index.indices(len(self)))
            ]
        else:
            start = index * self.itemsize
            end = start + self.itemsize
            return self._unpack(self.buffer[start:end])

    def _unpack(self, data):
        """
        Unpack a float16 value from bytes.

        Args:
            data (bytes): The bytes to unpack.

        Returns:
            float: The unpacked float16 value.
        """
        return struct.unpack(self.format, data)[0]

    def __len__(self):
        """
        Get the number of float16 values in the buffer.

        Returns:
            int: The number of float16 values.
        """
        return len(self.buffer) // self.itemsize

    def tolist(self):
        """
        Convert the buffer to a list of float16 values.

        Returns:
            list: The list of float16 values.
        """
        return [self[i] for i in range(len(self))]
