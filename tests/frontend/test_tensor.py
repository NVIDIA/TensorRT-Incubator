import inspect
import sys

import jax
import numpy as np
import pytest
import torch

import tripy as tp
from tripy.common.array import convert_tripy_to_module_dtype
from tripy.common.datatype import DATA_TYPES
from tripy.utils.stack_info import SourceInfo


class TestTensor:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = tp.Tensor(VALUES)

        assert isinstance(a, tp.Tensor)
        assert a.trace_tensor.producer.inputs == []
        assert isinstance(a.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        assert a.numpy().tolist() == VALUES

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_tensor_device(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))

        assert isinstance(a.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        assert a.trace_tensor.producer.device.kind == kind

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype(self, dtype):
        # (32): Allow setting all tripy supported types here.
        # Given a int/float data list, store data with requested data type.
        if dtype in {tp.int4, tp.bfloat16, tp.float8e4m3fn}:
            pytest.skip("Type is not supported by numpy/cupy")

        # (38): Add cast operation to support unsupported backend types. Allow requested type to be different than init data type for list data type.
        tensor = tp.Tensor(np.array([1, 2, 3], dtype=convert_tripy_to_module_dtype(dtype, np)))
        assert tensor.trace_tensor.producer.dtype == dtype
        assert tensor.trace_tensor.producer.data.dtype.name == dtype.name
        assert tensor.trace_tensor.producer.data.dtype.itemsize == dtype.itemsize

    # In this test we only check the two innermost stack frames since beyond that it's all pytest code.
    def test_stack_info_is_populated(self):
        class MockOp:
            def __init__(self, inputs, outputs):
                self.inputs = inputs
                self.outputs = outputs

            def infer_dtypes(self):
                pass

        # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
        expected_line_number = sys._getframe().f_lineno + 1
        a = tp.Tensor.build(inputs=[], OpType=lambda inputs, outputs: MockOp(inputs, outputs))

        # We don't check line number within tp.Tensor because it's diffficult to determine.
        assert a.stack_info[1] == SourceInfo(
            inspect.getmodule(tp.Tensor).__name__,
            file=inspect.getsourcefile(tp.Tensor),
            line=a.stack_info[1].line,
            function=tp.Tensor.build.__name__,
            code="",
            _dispatch_target="",
            column_range=None,
        )
        assert a.stack_info[2] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_line_number,
            function=TestTensor.test_stack_info_is_populated.__name__,
            code="        a = tp.Tensor.build(inputs=[], OpType=lambda inputs, outputs: MockOp(inputs, outputs))",
            _dispatch_target="",
            column_range=None,
        )

    def test_eval_of_storage_tensor_is_nop(self):
        a = tp.Tensor(np.array([1], dtype=np.float32))

        # TODO: Verify that we don't compile/execute somehow.
        assert a.numpy().tolist() == [1]

    def test_evaled_tensor_becomes_concrete(self):
        a = tp.Tensor(np.array([1], dtype=np.float32))
        b = tp.Tensor(np.array([2], dtype=np.float32))

        c = a + b
        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.BinaryElementwise)

        c.eval()

        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        # Storage tensors should have no inputs since we don't want to trace back from them.
        assert c.trace_tensor.producer.inputs == []
        assert (c.trace_tensor.producer.data.view().get() == np.array([3], dtype=np.float32)).all()

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_dlpack_torch(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))
        b = torch.from_dlpack(a)
        assert np.array_equal(a.numpy(), b.cpu().numpy())

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_dlpack_jax(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))
        b = jax.dlpack.from_dlpack(a)
        assert np.array_equal(a.numpy(), np.asarray(b))
