import pytest

import tripy as tp
from tripy import utils
from tripy.frontend.dim import dynamic_dim
from tests import helper
from collections import defaultdict


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


def make_with_constant_field():
    @utils.constant_fields("field")
    class WithConstField:
        def __init__(self):
            self.custom_setter_called_count = defaultdict(int)
            self.field = 0
            self.other_field = 1

        def __setattr__(self, name, value):
            if name != "custom_setter_called_count":
                self.custom_setter_called_count[name] += 1
            return super().__setattr__(name, value)

    return WithConstField()


@pytest.fixture()
def with_const_field():
    yield make_with_constant_field()


class TestConstantFields:
    def test_field_is_immuatable(self, with_const_field):
        with helper.raises(
            tp.TripyException, match="Field: 'field' of class: '[a-zA-Z<>._]+?WithConstField' is immutable"
        ):
            with_const_field.field = 1

    def test_does_not_affect_other_fields(self, with_const_field):
        with_const_field.other_field = 3

    def test_does_not_override_custom_setter(self, with_const_field):
        assert with_const_field.custom_setter_called_count["other_field"] == 1
        with_const_field.other_field = 2
        assert with_const_field.custom_setter_called_count["other_field"] == 2

    def test_is_per_instance(self):
        const0 = make_with_constant_field()
        # When we initialize the `field` value for the second instance, it should NOT fail due to
        # the first instance already having initialized the field. This could happen if the implementation
        # doesn't take the instance into account when checking if the field has been initialized.
        const1 = make_with_constant_field()
