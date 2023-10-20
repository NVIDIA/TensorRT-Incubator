import inspect
import sys

import pytest

from tripy.frontend import TensorExpression

# Internal-only imports
from tripy.util.stack_info import SourceInfo
import tripy.ops


class TestTensorExpression:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = TensorExpression.tensor(VALUES)

        assert isinstance(a, TensorExpression)
        assert a.inputs == []
        assert isinstance(a.op, tripy.ops.Value)
        assert a.op.values == VALUES

    # In this test we only check the two innermost stack frames since beyond that it's all pytest code.
    def test_stack_info_is_populated(self):
        # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
        expected_line_number = sys._getframe().f_lineno + 1
        a = TensorExpression(inputs=[], op=None)

        # We don't check line number within TensorExpression because it's diffficult to determine.
        assert a._stack_info[0] == SourceInfo(
            inspect.getmodule(TensorExpression).__name__,
            file=inspect.getsourcefile(TensorExpression),
            line=a._stack_info[0].line,
            function=TensorExpression.__init__.__name__,
        )
        assert a._stack_info[1] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_line_number,
            function=TestTensorExpression.test_stack_info_is_populated.__name__,
        )

    @pytest.mark.parametrize("func, kind", [(lambda a, b: a + b, tripy.ops.BinaryElementwise.Kind.SUM)])
    def test_binary_elementwise(self, func, kind):
        a = TensorExpression.tensor([1])
        b = TensorExpression.tensor([2])

        out = func(a, b)
        assert isinstance(out, TensorExpression)
        assert isinstance(out.op, tripy.ops.BinaryElementwise)
        assert out.op.kind == kind
