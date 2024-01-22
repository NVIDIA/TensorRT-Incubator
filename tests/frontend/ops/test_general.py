import inspect

import pytest

from tests import helper
from tripy.frontend.ops import BaseOperator

OP_TYPES = {obj for obj in helper.discover_tripy_objects() if inspect.isclass(obj) and issubclass(obj, BaseOperator)}


@pytest.mark.parametrize("OpType", OP_TYPES)
def test_to_flat_ir_does_not_access_io(OpType):
    # to_flat_ir() methods should not access `self.inputs` and `self.outputs`
    source = inspect.getsource(OpType.to_flat_ir)

    assert "self.inputs" not in source
    assert "self.outputs" not in source
