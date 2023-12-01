from typing import List
import ctypes
import cupy as cp
import tripy.common.datatype
from tripy.common.datatype import DataType

MAX_DIMS = 8
BIT_WIDTH = 8

# Common types
void_ptr = ctypes.c_void_p
char_ptr = ctypes.c_char_p
c_int = ctypes.c_int
POINTER = ctypes.POINTER

ShapeInfo = tuple[int, ...]


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


class Dims(ctypes.Structure):
    """
    Represents the dimensions of a tensor.

    Attributes:
        - nb_dims (int): Number of dimensions.
        - dims (ctypes.c_int * MAX_DIMS): Array storing the dimensions.
    """

    _fields_ = [("nb_dims", ctypes.c_int), ("dims", (ctypes.c_int * MAX_DIMS))]


class TensorShape(ctypes.Structure):
    """
    Represents the shape of a tensor in MLIR.

    Attributes:
        - dtype (MlirDataType): Data type of the tensor.
        - dims (Dims): Dimensions of the tensor.

    Methods:
        - get_mlir_dtype(): Get the data type as a tripy.common.datatype.
        - get_tripy_dtype(): Get the tripy.common.datatype equivalent of the data type.
        - get_cupy_dtype(): Get the cupy data type equivalent of the data type.
        - get_nb_elements(): Get the total number of elements in the tensor.
        - get_size_in_bytes(): Get the size of the tensor in bytes.
        - get_shape_arr(): Get the array of dimensions.
    """

    _fields_ = [("dtype", MlirDataType), ("dims", Dims)]

    def __init__(self, dtype: DataType = tripy.common.datatype.float32, shape: List = []):
        def _get_mlir_dtype(dtype: DataType) -> MlirDataType:
            """Get the data type of the tensor."""
            """Get the tripy.common.datatype equivalent of the data type."""
            if isinstance(dtype, MlirDataType):
                return dtype.value

            if isinstance(dtype, DataType):
                to_mlir = {
                    "float32": 0,
                    "float16": 1,
                    "int8": 2,
                    "int32": 3,
                    "bool": 4,
                    "uint8": 5,
                    "float8e4m3fn": 6,
                    "bfloat16": 7,
                    "int64": 8,
                    "int4": 9,
                }
                return to_mlir[dtype.name]

        self.dtype = _get_mlir_dtype(dtype)
        self.dims = Dims(len(shape), (ctypes.c_int * MAX_DIMS)(*shape))

    def get_mlir_dtype(self) -> MlirDataType:
        """Get the data type of the tensor."""
        return self.dtype.value

    def get_tripy_dtype(self) -> tripy.common.datatype:
        """Get the tripy.common.datatype equivalent of the data type."""
        to_tripy = {
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
        return to_tripy[self.get_mlir_dtype()]

    def get_cupy_dtype(self) -> cp.dtype:
        """Get the cupy data type equivalent of the data type."""
        to_cupy = {
            tripy.common.datatype.float32: cp.float32,
            tripy.common.datatype.float16: cp.float16,
            tripy.common.datatype.int8: cp.int8,
            tripy.common.datatype.int32: cp.int32,
            tripy.common.datatype.int64: cp.int64,
            tripy.common.datatype.uint8: cp.uint8,
        }
        return to_cupy[self.get_tripy_dtype()]

    def get_nb_elements(self) -> int:
        """Get the total number of elements in the tensor."""
        nb_elements = 1
        for i in range(self.dims.nb_dims):
            nb_elements *= self.dims.dims[i]
        return nb_elements

    def get_size_in_bytes(self) -> int:
        """Get the size of the tensor in bytes."""
        return self.get_nb_elements() * self.get_tripy_dtype().itemsize * BIT_WIDTH

    def get_shape_arr(self) -> List[int]:
        """Get the list of dimensions."""
        return list(self.dims.dims[: self.dims.nb_dims])
