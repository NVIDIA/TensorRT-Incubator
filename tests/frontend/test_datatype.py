import pytest

import tripy as tp
from tripy.common.datatype import DATA_TYPES


class TestDataType:
    @pytest.mark.parametrize("name", DATA_TYPES.keys())
    def test_api(self, name):
        # Make sure we can access data types at the top-level, e.g. `tripy.float32`
        assert isinstance(getattr(tp, name), tp.dtype)

    @pytest.mark.parametrize("name, dtype", DATA_TYPES.items())
    def test_name(self, name, dtype):
        assert name == dtype.name

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_itemsize(self, dtype):
        EXPECTED_ITEMSIZES = {
            "float32": 4,
            "float16": 2,
            "float8e4m3fn": 1,
            "bfloat16": 2,
            "int4": 0.5,
            "int8": 1,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
            "bool": 1,
        }
        assert dtype.itemsize == EXPECTED_ITEMSIZES[dtype.name]
