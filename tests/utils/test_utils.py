import pytest

import tripy as tp
from tripy import utils
from tripy.frontend.dim import dynamic_dim


class TestMd5:
    def test_hash_same_for_same_objects(self):
        assert utils.md5(0) == utils.md5(0)

    def test_hash_different_for_different_objects(self):
        assert utils.md5(0) != utils.md5(1)

    @pytest.mark.parametrize(
        "func",
        [
            # Check devices
            lambda: tp.device("cpu"),
            lambda: tp.device("cpu:4"),
            lambda: tp.device("gpu:1"),
        ],
    )
    def test_hash_equivalence(self, func):
        obj0 = func()
        obj1 = func()
        assert utils.md5(obj0) == utils.md5(obj1)


@pytest.mark.parametrize(
    "inp, expected",
    [
        ((2, 3, 4), (dynamic_dim(2), dynamic_dim(3), dynamic_dim(4))),
        (((2, dynamic_dim(3), 4)), (dynamic_dim(2), dynamic_dim(3), dynamic_dim(4))),
        (None, None),
        ((), ()),
    ],
)
def test_to_dims(inp, expected):
    assert utils.to_dims(inp) == expected
