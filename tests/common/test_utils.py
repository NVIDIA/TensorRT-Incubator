import pytest
import struct
from collections import ChainMap
from textwrap import dedent

import cupy as cp
import numpy as np
import jax.numpy as jnp
import torch

import tripy.common.datatype

from tests import helper

from tripy.common.exception import TripyException
from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype, convert_list_to_bytebuffer, get_element_type


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

    PYTHON_NATIVE_TO_TRIPY = {
        int: tripy.common.datatype.int32,
        float: tripy.common.datatype.float32,
        bool: tripy.common.datatype.bool,
    }

    CUPY_TO_TRIPY = {
        cp.bool_: tripy.common.datatype.bool,
        cp.int8: tripy.common.datatype.int8,
        cp.int32: tripy.common.datatype.int32,
        cp.int64: tripy.common.datatype.int64,
        cp.uint8: tripy.common.datatype.uint8,
        cp.float16: tripy.common.datatype.float16,
        cp.float32: tripy.common.datatype.float32,
    }

    NUMPY_TO_TRIPY = {
        np.bool_: tripy.common.datatype.bool,
        np.int8: tripy.common.datatype.int8,
        np.int32: tripy.common.datatype.int32,
        np.int64: tripy.common.datatype.int64,
        np.uint8: tripy.common.datatype.uint8,
        np.float16: tripy.common.datatype.float16,
        np.float32: tripy.common.datatype.float32,
    }

    TORCH_TO_TRIPY = {
        torch.bool: tripy.common.datatype.bool,
        torch.int8: tripy.common.datatype.int8,
        torch.int32: tripy.common.datatype.int32,
        torch.int64: tripy.common.datatype.int64,
        torch.uint8: tripy.common.datatype.uint8,
        torch.float16: tripy.common.datatype.float16,
        torch.bfloat16: tripy.common.datatype.bfloat16,
        torch.float32: tripy.common.datatype.float32,
    }

    JAX_TO_TRIPY = {
        jnp.bool_: tripy.common.datatype.bool,
        jnp.int8: tripy.common.datatype.int8,
        jnp.int32: tripy.common.datatype.int32,
        jnp.int64: tripy.common.datatype.int64,
        jnp.uint8: tripy.common.datatype.uint8,
        jnp.float8_e4m3fn: tripy.common.datatype.float8,
        jnp.float16: tripy.common.datatype.float16,
        jnp.bfloat16: tripy.common.datatype.bfloat16,
        jnp.float32: tripy.common.datatype.float32,
    }

    FRONTEND_TO_TRIPY = dict(ChainMap(PYTHON_NATIVE_TO_TRIPY, NUMPY_TO_TRIPY, TORCH_TO_TRIPY, JAX_TO_TRIPY))

    for frontend_type, tripy_type in FRONTEND_TO_TRIPY.items():
        assert convert_frontend_dtype_to_tripy_dtype(frontend_type) == tripy_type

    for unsupported_type in [
        "unsupported_type",
        np.int16,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float64,
        cp.int16,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float64,
        torch.int16,
        torch.float64,
        jnp.int4,
        jnp.int16,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.float64,
    ]:
        with helper.raises(
            TripyException,
            match=dedent(
                rf"""
            Unsupported data type: '{unsupported_type}'.
                Tripy tensors can be constructed with one of the following data types: int, float, bool, bool_, int8, int32, int64, uint8, float8_e4m3fn, float16, bfloat16, float32.
            """
            ).strip(),
        ):
            convert_frontend_dtype_to_tripy_dtype(unsupported_type)


@pytest.mark.parametrize(
    "values, dtype, expected",
    [
        ([True, False, True], tripy.common.datatype.bool, b"\x01\x00\x01"),
        ([1, 2, 3], tripy.common.datatype.int8, b"\x01\x02\x03"),
        ([100000, 200000], tripy.common.datatype.int32, b"\xa0\x86\x01\x00@\x0d\x03\x00"),
        ([1, 2], tripy.common.datatype.int64, b"\x01\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00"),
        ([1, 2, 3], tripy.common.datatype.uint8, b"\x01\x02\x03"),
        ([1.0, 2.0], tripy.common.datatype.float16, b"\x00<\x00@"),
        ([1.0, 2.0], tripy.common.datatype.float32, b"\x00\x00\x80?\x00\x00\x00@"),
    ],
)
def test_convert_list_to_bytebuffer(values, dtype, expected):
    result = convert_list_to_bytebuffer(values, dtype)
    assert result == expected
