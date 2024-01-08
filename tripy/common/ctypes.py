import ctypes
import cupy as cp
import numpy as np
from typing import List

import tripy.common.datatype

MAX_DIMS = 8

# Common types
void_ptr = ctypes.c_void_p
char_ptr = ctypes.c_char_p
c_int = ctypes.c_int
c_int64 = ctypes.c_int64
POINTER = ctypes.POINTER


class MlirDataType(ctypes.c_int):
    """
    Represents data types used in MLIR.

    Attributes:
        - float32
        - float16
        - int8
        - int32
        - bool
        - uint8
        - float8e4m3fn
        - bfloat16
        - int64
        - int4
    """

    _fields_ = []


def convert_mlirdtype_to_tripy_dtype(dtype: MlirDataType):
    ctypes_to_tripy = dict(
        {
            0: tripy.common.datatype.float32,
            1: tripy.common.datatype.float16,
            2: tripy.common.datatype.int8,
            3: tripy.common.datatype.int32,
            4: tripy.common.datatype.bool,
            5: tripy.common.datatype.uint8,
            6: tripy.common.datatype.float8e4m3fn,
            7: tripy.common.datatype.bfloat16,
            8: tripy.common.datatype.int64,
            9: tripy.common.datatype.int4,
        }
    )
    return ctypes_to_tripy[dtype.value]


class Dims(ctypes.Structure):
    """
    Represents the dimensions of a tensor.

    Attributes:
        - nb_dims (int): Number of dimensions.
        - dims (ctypes.c_int64 * MAX_DIMS): Array storing the dimensions.
    """

    _fields_ = [("nb_dims", ctypes.c_int), ("dims", (ctypes.c_int64 * MAX_DIMS))]


class TensorShape(ctypes.Structure):
    """
    Represents the shape of a tensor in MLIR.

    Attributes:
        - dtype (MlirDataType): Data type of the tensor.
        - dims (Dims): Dimensions of the tensor.

    Methods:
        - get_shape_arr(): Get the array of dimensions.
    """

    _fields_ = [("dtype", MlirDataType), ("dims", Dims)]

    def get_shape_arr(self) -> List[int]:
        """Get the list of dimensions."""
        return list(self.dims.dims[: self.dims.nb_dims])
