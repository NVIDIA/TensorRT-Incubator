import inspect
import sys

import numpy as np
import pytest

import tripy
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

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_tensor_device(self, kind):
        a = Tensor([1, 2, 3], device=tripy.device(kind))

        assert isinstance(a.op, tripy.ops.Storage)
        assert a.op.device.kind == kind

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

    def test_eval_of_storage_tensor_is_nop(self):
        a = Tensor(np.array([1], dtype=np.float32))

        # TODO: Verify that we don't compile/execute somehow.
        assert list(a.eval()) == [1]

    def test_evaled_tensor_becomes_concrete(self):
        a = Tensor(np.array([1], dtype=np.float32))
        b = Tensor(np.array([2], dtype=np.float32))

        c = a + b
        assert isinstance(c.op, tripy.ops.BinaryElementwise)

        assert list(c.eval()) == [3]

        assert isinstance(c.op, tripy.ops.Storage)
        # Storage tensors should have no inputs since we don't want to trace back from them.
        assert c.inputs == []
        assert list(c.op.data) == [3]
