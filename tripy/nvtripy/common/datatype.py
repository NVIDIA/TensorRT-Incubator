#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from nvtripy import export
from nvtripy.utils.json import Decoder, Encoder

# A dictionary to store data types
DATA_TYPES = {}
INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


# We include a metaclass here so we can control how the class type is printed.
# The default would be something like: `<class 'abc.float32'>`. With the metaclass,
# we can print it as `float32`
@export.public_api(document_under="datatype.rst")
class dtype(type):
    """
    The base type for all data types supported by nvtripy.
    """

    name: str
    """The human-readable name of the data type"""

    itemsize: float
    """The size of an element of this data type, in bytes"""

    def __str__(cls):
        return cls.name

    def __repr__(cls):
        return cls.name


class BaseDtype(metaclass=dtype):
    """
    The base class for all data types supported by nvtripy.
    """

    name = "BaseDtype"
    itemsize = -1


@export.public_api(document_under="datatype.rst", autodoc_options=[":no-show-inheritance:"])
class integer(BaseDtype):
    """
    The base class for all integer data types.
    """

    name = "integer"
    itemsize = -1


@export.public_api(document_under="datatype.rst", autodoc_options=[":no-show-inheritance:"])
class floating(BaseDtype):
    """
    The base class for all floating-point data types.
    """

    name = "floating"
    itemsize = -1


# We use `__all__` to control what is exported from this file. `import *` will only pull in objects that are in `__all__`.
__all__ = ["dtype"]


def _make_datatype(name, shortname, dtypeclass, itemsize, docstring):
    DATA_TYPES[name] = export.public_api(document_under="datatype.rst", autodoc_options=[":no-show-inheritance:"])(
        type(name, (dtypeclass,), {"name": name, "shortname": shortname, "itemsize": itemsize, "__doc__": docstring})
    )
    __all__.append(name)
    return DATA_TYPES[name]


def _make_float_datatype(name, shortname, itemsize, docstring):
    return _make_datatype(name, shortname, floating, itemsize, docstring)


def _make_int_datatype(name, shortname, itemsize, docstring):
    return _make_datatype(name, shortname, integer, itemsize, docstring)


float32 = _make_float_datatype("float32", "f32", 4, "32-bit floating point")
float16 = _make_float_datatype("float16", "f16", 2, "16-bit floating point")
float8 = _make_float_datatype("float8", "f8", 1, "8-bit floating point")
bfloat16 = _make_float_datatype("bfloat16", "bf16", 2, "16-bit brain-float")
int4 = _make_int_datatype("int4", "i4", 0.5, "4-bit signed integer")
int8 = _make_int_datatype("int8", "i8", 1, "8-bit signed integer")
int32 = _make_int_datatype("int32", "i32", 4, "32-bit signed integer")
int64 = _make_int_datatype("int64", "i64", 8, "64-bit signed integer")
bool = _make_datatype("bool", "b", BaseDtype, 1, "Boolean")


@Encoder.register(dtype)
def encode(obj: dtype) -> Dict[str, Any]:
    return {"name": obj.name}


@Decoder.register(dtype)
def decode(dct: Dict[str, Any]) -> dtype:
    return DATA_TYPES[dct["name"]]
