#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import jax
import numpy as np
import pytest
import torch

import tripy as tp
from tests.conftest import DATA_TYPE_TEST_CASES
from tests.helper import NUMPY_TYPES
from tripy.utils.stack_info import SourceInfo
from tripy.common.utils import get_element_type


class TestTensor:
    def test_tensor(self):
        VALUES = [1, 2, 3]
        a = tp.Tensor(VALUES)

        assert isinstance(a, tp.Tensor)
        assert a.trace_tensor.producer.inputs == []
        assert isinstance(a.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        assert cp.from_dlpack(a).get().tolist() == VALUES

    def test_empty_tensor(self):
        a = tp.Tensor([], dtype=tp.int32)  # dtype cannot be inferred for empty tensor
        assert isinstance(a, tp.Tensor)
        assert cp.from_dlpack(a).get().tolist() == []

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

    def test_bool_tensor(self):
        bool_values = [True, False, True]
        t = tp.Tensor(bool_values, dtype=tp.bool)
        assert isinstance(t, tp.Tensor)
        assert t.trace_tensor.producer.inputs == []
        assert cp.from_dlpack(t).get().tolist() == bool_values

    @pytest.mark.parametrize("input_data", [[], [0.0, 1.0, 2.0, 3.0], [1, 2, 3, 4], [False, True, False, True]])
    @pytest.mark.parametrize("dtype", DATA_TYPE_TEST_CASES)
    def test_dtype_from_list(self, input_data, dtype):
        if dtype == tp.int4:
            pytest.skip(f"Unsupported front-end data type {dtype}")
        # Error: input.dtype must be one of (float32, float16, bfloat16), Got dtype=bool
        if (get_element_type(input_data) in [tp.int32, tp.bool]) and dtype == tp.float8:
            pytest.skip(
                f"Input data {input_data} with {get_element_type(input_data)} type can not be implicitly converted to {dtype}"
            )
        # Error: 'plan.inline_closed_group' op input operand #0 of type 'tensor<0xf32>' does not have a TensorKind associated with it
        if len(input_data) == 0 and dtype == tp.float8:
            pytest.skip(
                f"Input data {input_data} with {get_element_type(input_data)} type can not be implicitly converted to {dtype}"
            )
        tensor = tp.Tensor(input_data, dtype=dtype)
        assert tensor.trace_tensor.producer.dtype == dtype
        if dtype == tp.float8:
            assert tensor.data().dtype == tp.float32
        else:
            assert tensor.data().dtype == dtype
        assert tensor.trace_tensor.producer.data.dtype.name == dtype.name
        assert tensor.trace_tensor.producer.data.dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", DATA_TYPE_TEST_CASES)
    def test_dtype_printing(self, dtype):
        if dtype == tp.int4:
            pytest.skip(f"Unsupported front-end data type {dtype}")
        from tripy.logging import logger

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
            code=None,
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
        a = tp.Tensor(cp.array([1], dtype=cp.float32))

        # TODO: Verify that we don't compile/execute somehow.
        assert cp.from_dlpack(a).get().tolist() == [1]

    def test_evaled_tensor_becomes_concrete(self):
        a = tp.Tensor(cp.array([1], dtype=cp.float32))
        b = tp.Tensor(cp.array([2], dtype=cp.float32))

        c = a + b
        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.BinaryElementwise)

        c.eval()

        assert isinstance(c.trace_tensor.producer, tp.frontend.trace.ops.Storage)
        # Storage tensors should have no inputs since we don't want to trace back from them.
        assert c.trace_tensor.producer.inputs == []
        assert (cp.from_dlpack(c.trace_tensor.producer.data) == cp.array([3], dtype=np.float32)).all()

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_dlpack_torch(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))
        b = torch.from_dlpack(a)
        assert torch.equal(b.cpu(), torch.tensor([1, 2, 3]))

    @pytest.mark.parametrize("kind", ["cpu", "gpu"])
    def test_dlpack_jax(self, kind):
        a = tp.Tensor([1, 2, 3], device=tp.device(kind))
        b = jax.dlpack.from_dlpack(a)
        assert jax.numpy.array_equal(b, jax.numpy.array([1, 2, 3]))

    def test_stack_depth_sanity(self):
        # Makes sure STACK_DEPTH_OF_BUILD is correct
        a = tp.ones((2, 3))

        def find_frame(func_name):
            for frame in a.stack_info:
                if frame.function == func_name:
                    return frame
            assert False, f"Could not find frame for function: {func_name}"

        # Make sure we include code for not only the `ones()` API but also the `full()` API
        # that it calls underneath
        assert find_frame("ones").code.strip() == "return full(shape, 1, dtype)"
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
