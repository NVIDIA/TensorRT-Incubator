import inspect

import pytest
import dataclasses
from tests import helper
from tripy.frontend.trace.ops import BaseTraceOp

OP_TYPES = {obj for obj in helper.discover_tripy_objects() if inspect.isclass(obj) and issubclass(obj, BaseTraceOp)}


@pytest.mark.parametrize("OpType", OP_TYPES)
class TestFrontendOps:
    def test_to_flat_ir_does_not_access_io(self, OpType):
        # to_flat_ir() methods should not access `self.inputs` and `self.outputs`
        source = inspect.getsource(OpType.to_flat_ir)

        assert "self.inputs" not in source
        assert "self.outputs" not in source

    def test_is_dataclass(self, OpType):
        assert dataclasses.is_dataclass(
            OpType
        ), f"Frontend ops must be data classes since many base implementations rely on dataclass introspection"

    def test_has_no_dataclass_repr(self, OpType):
        # If you define a custom repr, add a waiver here.
        assert (
            OpType.__repr__ is BaseTraceOp.__repr__
        ), "Use @dataclass(repr=False) to avoid extremely verbose __repr__ implementations"
