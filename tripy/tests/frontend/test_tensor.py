#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import inspect
import sys

import cupy as cp
import numpy as np
import nvtripy as tp
import pytest
import torch
from nvtripy.trace.ops.constant import Constant
from nvtripy.utils.stack_info import SourceInfo
from tests.conftest import DATA_TYPE_TEST_CASES
from tests.helper import NUMPY_TO_TRIPY


class TestTensor:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = tp.Tensor(VALUES)

        assert isinstance(a, tp.Tensor)
        assert a.trace_tensor.producer.inputs == []
        assert isinstance(a.trace_tensor.producer, Constant)
        assert cp.from_dlpack(a).get().tolist() == VALUES

    def test_empty_tensor(self):
        a = tp.Tensor([], dtype=tp.int32)  # dtype cannot be inferred for empty tensor
        assert isinstance(a, tp.Tensor)
        assert cp.from_dlpack(a).get().tolist() == []

    def test_input_list_is_copied(self):
        # Make sure that if we initialize the tensor with a list, the tensor
        # contents are not modified if we update the list.
        lst = [1, 2, 3]
        a = tp.Tensor(lst, dtype=tp.int32)

        lst.clear()
        assert lst == []
        assert a.tolist() == [1, 2, 3]

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_tensor_device(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))

        assert isinstance(a.trace_tensor.producer, Constant)
        assert a.trace_tensor.producer.device.kind == kind

    @pytest.mark.parametrize("dtype", NUMPY_TO_TRIPY.keys())
    def test_dtype_from_numpy(self, dtype):

        np_array = np.array([1, 2, 3], dtype=dtype)
        tensor = tp.Tensor(np_array)
        tp_dtype = NUMPY_TO_TRIPY[dtype]
        assert tensor.dtype == tp_dtype

    def test_bool_tensor(self):
        bool_values = [True, False, True]
        t = tp.Tensor(bool_values, dtype=tp.bool)
        assert isinstance(t, tp.Tensor)
        assert t.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(t).get().tolist() == bool_values

    @pytest.mark.parametrize("input_data", [[], [0.0, 1.0, 2.0, 3.0], [1, 2, 3, 4], [False, True, False, True]])
    @pytest.mark.parametrize("dtype", DATA_TYPE_TEST_CASES)
    def test_dtype_from_list(self, input_data, dtype):
        tensor = tp.Tensor(input_data, dtype=dtype)
        assert tensor.dtype == dtype

    @pytest.mark.parametrize("dtype", DATA_TYPE_TEST_CASES)
    def test_dtype_printing(self, dtype):
        if dtype == tp.int4:
            pytest.skip(f"Unsupported frontend data type: {dtype}")

        # This is required to print intermediate data representations.
        with tp.logger.use_verbosity("ir"):
            data = [0.0, 1.0, 2.0, 3.0]
            if dtype == tp.bool:
                data = [False, True, False, True]
            elif dtype in [tp.int8, tp.int32, tp.int64]:
                data = [0, 1, 2, 3]
            a = tp.Tensor(data, dtype=dtype)
            print(a)

    # In this test we only check the two innermost stack frames since beyond that it's all pytest code.
    @pytest.mark.parametrize(
        "build_func,expected_line_number,expected_func",
        [
            (lambda: tp.Tensor([1, 1, 1]), sys._getframe().f_lineno, tp.Tensor.__init__),
            (lambda: tp.ones((3,)), sys._getframe().f_lineno, tp.Tensor.from_trace_tensor),
        ],
        ids=["constructor", "op"],
    )
    def test_stack_info_is_populated(self, build_func, expected_line_number, expected_func):
        a = build_func()
        a.stack_info.fetch_source_code()

        assert a.stack_info[0] == SourceInfo(
            inspect.getmodule(tp.Tensor).__name__,
            file=inspect.getsourcefile(tp.Tensor),
            # We don't check line number within tp.Tensor because it's difficult to determine.
            line=a.stack_info[0].line,
            function=expected_func.__name__,
            code=None,
            _dispatch_target="",
            column_range=(25, 30) if sys.version_info >= (3, 11) else None,
        )
        assert a.stack_info[a.stack_info.get_first_user_frame_index()] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_line_number,
            function=build_func.__name__,
            code=inspect.getsource(build_func).rstrip(),
            _dispatch_target="",
            column_range=(0, 0) if sys.version_info >= (3, 11) else None,
        )

    def test_eval_of_storage_tensor_is_nop(self):
        a = tp.Tensor(cp.array([1], dtype=cp.float32))

        # TODO: Verify that we don't compile/execute somehow.
        assert cp.from_dlpack(a).get().tolist() == [1]

    def test_evaled_tensor_becomes_concrete(self):
        a = tp.Tensor(cp.array([1], dtype=cp.float32))
        b = tp.Tensor(cp.array([2], dtype=cp.float32))

        c = a + b
        assert not isinstance(c.trace_tensor.producer, Constant)

        c.eval()

        assert isinstance(c.trace_tensor.producer, Constant)
        # Constant tensors should have no inputs since we don't want to trace back from them.
        assert c.trace_tensor.producer.inputs == []
        assert (cp.from_dlpack(c.trace_tensor.producer.data) == cp.array([3], dtype=np.float32)).all()

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_dlpack_torch(self, kind):
        a_torch = torch.ones((2, 2), dtype=torch.float32)
        if kind == "gpu":
            a_torch = a_torch.to("cuda")
        assert torch.equal(a_torch, torch.from_dlpack(tp.Tensor(a_torch)))

    def test_stack_depth_sanity(self):
        a = tp.ones((2, 3))
        a.stack_info.fetch_source_code()

        def find_frame(func_name):
            for frame in a.stack_info:
                if frame.function == func_name:
                    return frame
            assert False, f"Could not find frame for function: {func_name}"

        # Make sure we include code for not only the `ones()` API but also the `full()` API
        # that it calls underneath
        assert find_frame("ones").code.strip() == "return full(shape, 1.0, dtype)"
        assert find_frame("test_stack_depth_sanity").code.strip() == "a = tp.ones((2, 3))"

    @pytest.mark.parametrize(
        "tensor",
        [
            tp.Tensor([0]),
            tp.Tensor([1]),
            tp.zeros((1, 1, 1)),
            tp.ones((1, 1, 1)),
            tp.Tensor([[[3.12]]]),
            tp.Tensor([False]),
            tp.Tensor([True]),
        ],
    )
    def test_boolean_method(self, tensor):
        assert bool(tensor) == bool(cp.from_dlpack(tensor))

    def test_multiple_elements_boolean_fails(self):
        tensor = tp.ones((2, 2))

        with pytest.raises(tp.TripyException):
            bool(tensor)

    @pytest.mark.parametrize(
        "data",
        [
            [[1, 2], [3, 4]],  # from python list
            np.ones((2, 2), dtype=np.float32),  # from ext tensor
        ],
    )
    def test_explicit_cast(self, data):
        a = tp.Tensor(data, dtype=tp.float16)
        assert a.dtype == tp.float16

    def test_no_explicit_cast(self):
        from nvtripy.trace.ops.constant import Constant

        a_np = np.ones((2, 2), dtype=np.float32)
        a = tp.Tensor(a_np, dtype=tp.float32)
        assert a.dtype == tp.float32
        assert isinstance(a.trace_tensor.producer, Constant)

    @pytest.mark.parametrize(
        "devices",
        [
            ("cpu", "gpu"),
            ("gpu", "cpu"),
        ],
    )
    def test_explicit_copy(self, devices):
        a_torch = torch.ones((2, 2), dtype=torch.float32)
        if devices[0] == "gpu":
            a_torch = a_torch.to("cuda")
        a = tp.Tensor(a_torch, device=tp.device(devices[1]))
        assert a.device.kind == devices[1]

    @pytest.mark.parametrize(
        "devices",
        [
            ("cpu", "cpu"),
            ("gpu", "gpu"),
        ],
    )
    def test_no_explicit_copy(self, devices):
        from nvtripy.trace.ops.constant import Constant

        a_torch = torch.ones((2, 2), dtype=torch.float32)
        if devices[0] == "gpu":
            a_torch = a_torch.to("cuda")
        a = tp.Tensor(a_torch, device=tp.device(devices[1]))
        assert a.device.kind == devices[1]
        assert isinstance(a.trace_tensor.producer, Constant)

    def test_explicit_cast_copy(self):
        a_np = np.ones((2, 2), dtype=np.float32)
        a = tp.Tensor(a_np, dtype=tp.float16, device=tp.device("gpu"))
        assert a.dtype == tp.float16
        assert a.device.kind == "gpu"

    @pytest.mark.parametrize(
        "tensor, expected",
        [
            (tp.Tensor([0]), [0]),
            (tp.zeros((1, 1, 1)), [[[0]]]),
            (tp.Tensor([[[0.1]]]), [[[0.1]]]),
            (tp.Tensor([True]), [True]),
            (tp.ones((1, 2), dtype=tp.float16), [[1.0, 1.0]]),
        ],
    )
    def test_tolist(self, tensor, expected):
        assert np.allclose(tensor.tolist(), expected)

    # testing the invariant that stack trace of build is not past the limit
    @pytest.mark.parametrize(
        "tensor",
        [
            tp.ones((2, 2)),
            # Slice is an interesting case because it adds slice_helper to the stack.
            # Additionally, the use of slices may also require more ops, increasing the total stack depth.
            (tp.Tensor([1, 2, 3]) + tp.Tensor([4, 5, 6]))[:],
            (tp.Tensor([1, 2, 3]) + tp.Tensor([4, 5, 6]))[0:],
            (tp.Tensor([1, 2, 3]) + tp.Tensor([4, 5, 6]))[:3],
            (tp.Tensor([1, 2, 3]) + tp.Tensor([4, 5, 6]))[0:3:1],
            (tp.Tensor([[1], [2], [3]]) + tp.Tensor([[4], [5], [6]]))[0],
        ],
    )
    def test_stack_depth_of_create_op(self, tensor):
        tensor.stack_info.fetch_source_code()

        # Ensure that we do not include code for any frame until after the caller of `op_utils.create_op`
        create_op_caller = len(tensor.stack_info)
        for index, source_info in enumerate(tensor.stack_info):
            if source_info.function == "create_op":
                create_op_caller = index + 1
                break

        for index, source_info in enumerate(tensor.stack_info):
            # Once we reach user code we can stop checking
            if source_info.file == __file__:
                assert source_info.code is not None
                break

            # We should include code starting one frame past the *caller* of `create_op`, i.e. we
            # should not see a call to `create_op` in the code stack trace we display.
            if index > create_op_caller:
                assert source_info.code is not None
            else:
                assert source_info.code is None
