import dataclasses
import inspect

import pytest

from tests import helper
from tripy.flat_ir.ops import BaseFlatIROp

OP_TYPES = {obj for obj in helper.discover_tripy_objects() if inspect.isclass(obj) and issubclass(obj, BaseFlatIROp)}


@pytest.mark.parametrize("OpType", OP_TYPES)
class TestFlatIROps:
    def test_is_dataclass(self, OpType):
        assert dataclasses.is_dataclass(
            OpType
        ), f"FlatIR ops must be data classes since many base implementations rely on dataclass introspection"

    def test_has_no_dataclass_repr(self, OpType):
        # If you define a custom repr, add a waiver here.
        assert (
            OpType.__repr__ is BaseFlatIROp.__repr__
        ), "Use @dataclass(repr=False) to avoid extremely verbose __repr__ implementations"
