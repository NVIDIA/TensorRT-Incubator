import inspect
import sys


import tripy.ops
from tripy.frontend import Tensor
from tripy.util.stack_info import SourceInfo


class TestTensor:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = Tensor(VALUES)

        assert isinstance(a, Tensor)
        assert a.inputs == []
        assert isinstance(a.op, tripy.ops.Storage)
        assert list(a.op.data) == VALUES

    # In this test we only check the two innermost stack frames since beyond that it's all pytest code.
    def test_stack_info_is_populated(self):
        # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
        expected_line_number = sys._getframe().f_lineno + 1
        a = Tensor.build(inputs=[], op=None)

        # We don't check line number within Tensor because it's diffficult to determine.
        assert a._stack_info[1] == SourceInfo(
            inspect.getmodule(Tensor).__name__,
            file=inspect.getsourcefile(Tensor),
            line=a._stack_info[1].line,
            function=Tensor.build.__name__,
        )
        assert a._stack_info[2] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_line_number,
            function=TestTensor.test_stack_info_is_populated.__name__,
        )
