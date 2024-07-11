from typing import Any, List, Optional, Tuple, Union

from tripy.common.exception import raise_error
import tripy.common.datatype


def is_supported_array_type(dtype: "tripy.common.datatype.dtype") -> bool:
    if dtype is None:
        return True  # If an Tensor is created from a list or number without dtype field set.
    return dtype in [
        tripy.common.datatype.float32,
        tripy.common.datatype.int32,
        tripy.common.datatype.int64,
        tripy.common.datatype.bool,
    ]


def get_element_type(elements):
    e = elements
    while (isinstance(e, List) or isinstance(e, tuple)) and len(e) > 0:
        e = e[0]
    if isinstance(e, bool):
        return tripy.common.datatype.bool
    if isinstance(e, int):
        return tripy.common.datatype.int32
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

    PYTHON_NATIVE_MAPPING = {
        int: tripy.common.datatype.int32,
        float: tripy.common.datatype.float32,
        bool: tripy.common.datatype.bool,
    }
    if dtype in PYTHON_NATIVE_MAPPING:
        return PYTHON_NATIVE_MAPPING[dtype]

    try:
        dtype_name = dtype.name
    except AttributeError:
        dtype_name = str(dtype).split(".", 1)[-1].strip("'>")

    # TODO(#182): Use DLPack/buffer protocol to convert FW types to MemRefValue.
    NUMPY_TO_TRIPY = {
        "int4": tripy.common.datatype.int4,
        "int8": tripy.common.datatype.int8,
        "int16": tripy.common.datatype.int16,
        "int32": tripy.common.datatype.int32,
        "int64": tripy.common.datatype.int64,
        "uint8": tripy.common.datatype.uint8,
        "float16": tripy.common.datatype.float16,
        "float32": tripy.common.datatype.float32,
        "float64": tripy.common.datatype.float64,
        "bool": tripy.common.datatype.bool,
    }

    return NUMPY_TO_TRIPY.get(dtype_name, None)
