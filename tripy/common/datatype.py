import abc
import numpy as np
import torch
from typing import Any, List
from tripy.util.util import StrictKeyTypeDict

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


class DataTypeConverter:
    TRIPY_TO_NUMPY = StrictKeyTypeDict(
        {
            float32: np.float32,
            int32: np.int32,
            int8: np.int8,
            int64: np.int64,
            uint8: np.uint8,
            float16: np.float16,
        }
    )

    NUMPY_TO_TRIPY = StrictKeyTypeDict(
        {
            "int8": int8,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "float16": float16,
            "float32": float32,
            "bool": bool,
        }
    )

    @classmethod
    def convert_tripy_to_numpy_dtype(cls, dtype: Any) -> np.dtype:
        """
        Get the tripy.common.datatype equivalent of the data type.
        """
        return cls.TRIPY_TO_NUMPY[dtype]

    @classmethod
    def convert_numpy_to_tripy_dtype(cls, dtype: Any) -> Any:
        """
        Get the tripy.common.datatype equivalent of the data type.
        """
        if isinstance(dtype, torch.dtype):
            dtype_name = str(dtype).split(".", 1)[-1]
        elif any(isinstance(dtype, type(d)) for d in [int, float]):
            return float32 if dtype == float else int32
        else:
            dtype_name = dtype.name
        return cls.NUMPY_TO_TRIPY.get(dtype_name, None)
