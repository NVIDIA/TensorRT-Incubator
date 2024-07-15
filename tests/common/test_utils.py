import pytest
import numpy as np
from textwrap import dedent

import tripy.common.datatype

from tests import helper

from tripy.common.exception import TripyException
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype, is_supported_array_type, get_element_type


def test_is_supported_array_type():
    supported_dtype = [
        None,
        tripy.common.datatype.float32,
        tripy.common.datatype.int32,
        tripy.common.datatype.int64,
        tripy.common.datatype.bool,
    ]
    for dtype in supported_dtype:
        assert is_supported_array_type(dtype) is True

    unsupported_dtype = [
        tripy.common.datatype.int4,
        tripy.common.datatype.int8,
        tripy.common.datatype.uint8,
        tripy.common.datatype.float16,
        tripy.common.datatype.float8,
        tripy.common.datatype.bfloat16,
    ]
    for dtype in unsupported_dtype:
        assert is_supported_array_type(dtype) is False


def test_get_element_type():
    assert get_element_type([1, 2, 3]) == tripy.common.datatype.int32
    assert get_element_type([1.0, 2.0, 3.0]) == tripy.common.datatype.float32
    assert get_element_type([[1], [2], [3]]) == tripy.common.datatype.int32

    with helper.raises(
        TripyException,
        match="Unsupported element type.",
    ):
        get_element_type(["a", "b", "c"])


def test_convert_frontend_dtype_to_tripy_dtype():
    import numpy as np

    assert convert_frontend_dtype_to_tripy_dtype(tripy.common.datatype.int32) == tripy.common.datatype.int32

    FRONTEND_TYPE_TO_TRIPY = {
        int: tripy.common.datatype.int32,
        float: tripy.common.datatype.float32,
        bool: tripy.common.datatype.bool,
        np.int8: tripy.common.datatype.int8,
        np.int32: tripy.common.datatype.int32,
        np.int64: tripy.common.datatype.int64,
        np.uint8: tripy.common.datatype.uint8,
        np.float16: tripy.common.datatype.float16,
        np.float32: tripy.common.datatype.float32,
    }

    for frontend_type, tripy_type in FRONTEND_TYPE_TO_TRIPY.items():
        assert convert_frontend_dtype_to_tripy_dtype(frontend_type) == tripy_type

    for unsupported_type in ["unsupported_type", np.int16, np.uint16, np.uint32, np.uint64, np.float64]:
        # dtype_name = str(unsupported_type).split(".", 1)[-1].strip("'>")
        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Unsupported data type: '{unsupported_type}'.
                Tripy tensors can be constructed from arrays with one of the following data types: int4, int8, int32, int64, uint8, float16, float32, bfloat16, bool.
            """
            ).strip(),
        ):
            convert_frontend_dtype_to_tripy_dtype(unsupported_type)
