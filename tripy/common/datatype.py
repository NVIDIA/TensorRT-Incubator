from typing import Any, Dict

from tripy import export
from tripy.utils.json import Decoder, Encoder

# A dictionary to store data types
DATA_TYPES = {}


# We include a metaclass here so we can control how the class type is printed.
# The default would be something like: `<class 'abc.float32'>`. With the metaclass,
# we can print it as `float32`
@export.public_api(document_under="datatype.rst")
class dtype(type):
    """
    The base type for all data types supported by tripy.
    """

    def __str__(cls):
        return cls.name

    def __repr__(cls):
        return cls.name


class BaseDtype(metaclass=dtype):
    """
    The base class for all data types supported by tripy.
    """

    name = ""


# We use `__all__` to control what is exported from this file. `import *` will only pull in objects that are in `__all__`.
__all__ = ["dtype"]


def _make_datatype(name, itemsize, docstring):
    DATA_TYPES[name] = export.public_api(
        document_under="datatype.rst", autodoc_options=[":no-show-inheritance:"], include_heading=False
    )(type(name, (BaseDtype,), {"name": name, "itemsize": itemsize, "__doc__": docstring}))
    __all__.append(name)
    return DATA_TYPES[name]


float32 = _make_datatype("float32", 4, "32-bit floating point")
float16 = _make_datatype("float16", 2, "16-bit floating point")
float8e4m3fn = _make_datatype("float8e4m3fn", 1, "8-bit floating point")
bfloat16 = _make_datatype("bfloat16", 2, "16-bit brain-float")
int4 = _make_datatype("int4", 0.5, "4-bit signed integer")
int8 = _make_datatype("int8", 1, "8-bit signed integer")
int32 = _make_datatype("int32", 4, "32-bit signed integer")
int64 = _make_datatype("int64", 8, "64-bit signed integer")
uint8 = _make_datatype("uint8", 1, "8-bit unsigned integer")
bool = _make_datatype("bool", 1, "Boolean")


@Encoder.register(dtype)
def encode(obj: dtype) -> Dict[str, Any]:
    return {"name": obj.name}


@Decoder.register(dtype)
def decode(dct: Dict[str, Any]) -> dtype:
    return DATA_TYPES[dct["name"]]
