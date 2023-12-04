import abc

# A dictionary to store data types
DATA_TYPES = {}


class DataType(abc.ABC):
    pass


# We use `__all__` to control what is exported from this file. `import *` will only pull in objects that are in `__all__`.
__all__ = []


def _make_datatype(name, itemsize, docstring):
    DATA_TYPES[name] = type(name, (DataType,), {"name": name, "itemsize": itemsize, "__doc__": docstring})
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
