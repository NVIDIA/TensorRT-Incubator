import pytest

import tripy
from tripy.common.exception import TripyException


class TestDevice:
    def test_basic_construction(self):
        device = tripy.device("cpu")
        assert device.kind == "cpu"
        assert device.index == 0

    def test_index_construction(self):
        device = tripy.device("gpu:1")
        assert device.kind == "gpu"
        assert device.index == 1

    def test_invalid_device_kind_is_rejected(self):
        with pytest.raises(TripyException, match="Unrecognized device kind"):
            tripy.device("not_a_real_device_kind")

    def test_negative_device_index_is_rejected(self):
        with pytest.raises(TripyException, match="Device index must be a non-negative integer"):
            tripy.device("cpu:-1")

    def test_non_integer_device_index_is_rejected(self):
        with pytest.raises(TripyException, match="Could not interpret"):
            tripy.device("cpu:hi")
