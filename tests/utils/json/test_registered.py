import numpy as np
import pytest

from tripy.utils.json import from_json, to_json


class TestImplementations:
    @pytest.mark.parametrize(
        "obj",
        [
            np.ones((3, 4, 5), dtype=np.uint8),
        ],
        ids=lambda x: type(x),
    )
    def test_serde(self, obj):
        encoded = to_json(obj)
        decoded = from_json(encoded)
        if isinstance(obj, np.ndarray):
            assert np.array_equal(decoded, obj)
        else:
            assert decoded == obj
