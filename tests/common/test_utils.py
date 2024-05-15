import tripy.common.datatype

from tests import helper

from tripy.common.exception import TripyException
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype, is_supported_array_type, get_element_type


def test_is_supported_array_type():
    assert is_supported_array_type(None) is True
    assert is_supported_array_type(tripy.common.datatype.float32) is True
    assert is_supported_array_type(tripy.common.datatype.int32) is True
    assert is_supported_array_type(tripy.common.datatype.int64) is True
    assert is_supported_array_type(tripy.common.datatype.int8) is False


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

    assert convert_frontend_dtype_to_tripy_dtype(int) == tripy.common.datatype.int32
    assert convert_frontend_dtype_to_tripy_dtype(float) == tripy.common.datatype.float32

    assert convert_frontend_dtype_to_tripy_dtype(np.int32) == tripy.common.datatype.int32
    assert convert_frontend_dtype_to_tripy_dtype(np.float32) == tripy.common.datatype.float32

    assert convert_frontend_dtype_to_tripy_dtype("unsupported_type") is None
