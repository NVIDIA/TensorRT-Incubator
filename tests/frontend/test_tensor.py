import inspect
import sys

import jax
import numpy as np
import pytest
import torch

import tripy as tp
from tripy.common.datatype import DATA_TYPES
from tests.helper import NUMPY_TYPES
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

    @pytest.mark.parametrize("dtype", NUMPY_TYPES)
    def test_dtype_from_numpy(self, dtype):
        from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype

        np_array = np.array([1, 2, 3], dtype=dtype)
        tensor = tp.Tensor(np_array)
        tp_dtype = convert_frontend_dtype_to_tripy_dtype(dtype)
        assert tensor.trace_tensor.producer.dtype == tp_dtype
        assert tensor.trace_tensor.producer.data.dtype.name == tp_dtype.name
        assert tensor.trace_tensor.producer.data.dtype.itemsize == tp_dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_dtype_from_list(self, dtype):
        # Given a int/float data list, store data with requested data type.
        if dtype == tp.int4:
            pytest.skip("Int4 is not supported by frontend tensor.")
        if dtype == tp.bfloat16 and torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("bfloat16 requires GPU >= SM80")
        if dtype == tp.float8 and torch.cuda.get_device_capability() < (8, 9):
            pytest.skip("fp8 requires GPU >= SM89")
        # dtype casting is allowed for python list
        tensor = tp.Tensor([1, 2, 3], dtype=dtype)
        if dtype == tp.bool or dtype == tp.uint8:
            assert tensor.trace_tensor.producer.dtype == tp.int8
            assert tensor.trace_tensor.producer.data.dtype.name == "int8"
        else:
            assert tensor.trace_tensor.producer.dtype == dtype
        assert tensor.trace_tensor.producer.data.dtype.itemsize == dtype.itemsize

    # In this test we only check the two innermost stack frames since beyond that it's all pytest code.
    @pytest.mark.parametrize(
        "build_func,expected_line_number",
        [
            (lambda: tp.Tensor([1, 1, 1]), sys._getframe().f_lineno),
            (lambda: tp.ones((3,)), sys._getframe().f_lineno),
        ],
        ids=["constructor", "op"],
    )
    def test_stack_info_is_populated(self, build_func, expected_line_number):
        a = build_func()

        assert a.stack_info[0] == SourceInfo(
            inspect.getmodule(tp.Tensor).__name__,
            file=inspect.getsourcefile(tp.Tensor),
            # We don't check line number within tp.Tensor because it's difficult to determine.
            line=a.stack_info[0].line,
            function=tp.Tensor.__init__.__name__,
            code="",
            _dispatch_target="",
            column_range=None,
        )
        assert a.stack_info[a.stack_info.get_first_user_frame_index()] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_line_number,
            function=build_func.__name__,
            code=inspect.getsource(build_func).rstrip(),
            _dispatch_target="",
            column_range=None,
        )

    def test_eval_of_storage_tensor_is_nop(self):
        a = tp.Tensor(np.array([1], dtype=np.float32))

        # TODO: Verify that we don't compile/execute somehow.
        assert a.numpy().tolist() == [1]

    def test_evaled_tensor_becomes_concrete(self):
        import cupy as cp

        a = tp.Tensor(np.array([1], dtype=np.float32))
        b = tp.Tensor(np.array([2], dtype=np.float32))

        c = a + b
        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.BinaryElementwise)

        c.eval()

        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        # Storage tensors should have no inputs since we don't want to trace back from them.
        assert c.trace_tensor.producer.inputs == []
        assert (cp.from_dlpack(c.trace_tensor.producer.data).get() == np.array([3], dtype=np.float32)).all()

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
