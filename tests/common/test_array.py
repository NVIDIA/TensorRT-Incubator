import numpy as np
import cupy as cp
import pytest

from tripy.common.device import device
from tripy.common.datatype import int32, float32
from tripy.common.array import Array


@pytest.fixture
def int_data():
    return [1, 2, 3]


@pytest.fixture
def float_data():
    return [1.0, 2.0, 3.0]


def test_array_init_cpu(int_data):
    a = Array(int_data, int32, device("cpu"))
    assert isinstance(a, Array)
    assert a.device.kind == "cpu"


def test_array_init_gpu(int_data):
    a = Array(cp.array(int_data), int32, device("gpu"))
    assert isinstance(a, Array)
    assert a.device.kind == "gpu"


def test_array_view_cpu(float_data):
    a = Array(float_data, float32, device("cpu"))
    b = a.view(float32)

    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float32
    assert np.array_equal(b, np.array(float_data, dtype=np.float32))


def test_array_view_gpu(float_data):
    a = Array(cp.array(float_data), float32, device("gpu"))
    b = a.view(float32)

    assert isinstance(b, np.ndarray)
    assert b.dtype == np.float32
    assert np.array_equal(b, np.array(float_data, dtype=np.float32))


def test_array_equality(int_data):
    a = Array(int_data, int32, device("cpu"))
    b = Array(int_data, int32, device("cpu"))
    c = Array(cp.array(int_data), int32, device("gpu"))

    assert a == b
    assert a != c
    assert b != c
