import inspect
import sys

import numpy as np
import pytest

import tripy
import tripy.ops
from tripy.common.datatype import DATA_TYPES
from tripy.frontend import Tensor
from tripy.util.stack_info import SourceInfo


class TestTensor:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = Tensor(VALUES)

        assert isinstance(a, Tensor)
        assert a.inputs == []
        assert isinstance(a.op, tripy.ops.Storage)
        assert a.op.data.view(tripy.common.datatype.int32).tolist() == VALUES

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_tensor_device(self, kind):
        a = Tensor([1, 2, 3], device=tripy.device(kind))

        assert isinstance(a.op, tripy.ops.Storage)
        assert a.op.device.kind == kind

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.
        if dtype in {tripy.int4, tripy.bfloat16, tripy.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        if dtype in {tripy.int8, tripy.int64, tripy.float16, tripy.uint8, tripy.bool}:
            pytest.skip("Skip test until cast operation implemented.")

        data = [1, 2, 3] if dtype == tripy.int32 else [1.0, 2.0, 3.0]
        tensor = Tensor(data, dtype=dtype)
        assert tensor.op.dtype == dtype
        assert tensor.op.dtype.name == dtype.name
        assert tensor.op.dtype.itemsize == dtype.itemsize

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
        a = Tensor([1.0])

        # TODO: Verify that we don't compile/execute somehow.
        assert a.eval() == [1.0]

    def test_evaled_tensor_becomes_concrete(self):
        a = Tensor([1.0])
        b = Tensor([2.0])

        c = a + b
        assert isinstance(c.op, tripy.ops.BinaryElementwise)

        assert c.eval() == [3.0]

        assert isinstance(c.op, tripy.ops.Storage)
        # Storage tensors should have no inputs since we don't want to trace back from them.
        assert c.inputs == []
        # Replace with byte buffer check here.
        assert c.op.data.view(tripy.common.datatype.float32).tolist() == [3.0]
