#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import cupy as cp
import pytest
from tests import helper
from tests.backend.api.conftest import *

import nvtripy as tp
from nvtripy.frontend.trace.ops.storage import Storage


class TestCompile:
    # TODO (#246): Verify that it's actually compiling somehow here and below.
    # Need to return something programatically queriable from compile to do this.
    def test_function(self):
        compiled_relu = tp.compile(tp.relu, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        inp = tp.iota((2, 2), dtype=tp.float32) - 1
        out = compiled_relu(inp)

        assert tp.equal(out, tp.relu(inp))

    def test_module(self):
        layernorm = tp.LayerNorm(2)
        compiled_layernorm = tp.compile(layernorm, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        inp = tp.iota((2, 2), dtype=tp.float32) - 1
        out = compiled_layernorm(inp)

        assert tp.equal(out, layernorm(inp))

    def test_can_compile_using_shape_of_tensor(self):
        # Since InputInfo allows `DimensionSize`s, we should be able to use the shape of a tensor as
        # the shape of the InputInfo.
        inp = tp.iota((2, 2), dtype=tp.float32) - 1
        shape = inp.shape

        compiled_relu = tp.compile(tp.relu, args=[tp.InputInfo(shape, inp.dtype)])
        out = compiled_relu(inp)
        assert tp.equal(out, tp.relu(inp))

    def test_compile_arg_order_irrelevant(self):
        # The order of arguments we specify to `compile` should not affect the order
        # of the arguments in the compiled function, which should just follow the order
        # of the original function.
        compiled_sub = tp.compile(
            sub, kwargs=dict(b=tp.InputInfo((2, 2), dtype=tp.float32), a=tp.InputInfo((2, 2), dtype=tp.float32))
        )

        a = tp.ones((2, 2), dtype=tp.float32) * 2
        b = tp.ones((2, 2), dtype=tp.float32)

        # Compiled function should still take arguments in (a, b) order.
        out = compiled_sub(a, b)
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32))

    @pytest.mark.parametrize("b", [2, tp.ones((2, 2), dtype=tp.float32) * 2])
    def test_constants_baked(self, b):
        # Any non-InputInfo argument to compile is baked into the compiled function.
        compiled_add = tp.compile(add, args=[tp.InputInfo((2, 2), dtype=tp.float32), b])

        a = tp.zeros((2, 2), dtype=tp.float32)

        out = compiled_add(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32) * 2)

    @pytest.mark.parametrize("func", [variadic_positional, variadic_keyword])
    def test_variadic_arguments_rejected(self, func):
        with helper.raises(tp.TripyException, "Variadic positional/keyword arguments are not currently supported."):
            tp.compile(func)

    @pytest.mark.parametrize("func", [returns_non_tensor, returns_nothing])
    def test_invalid_return_rejected(self, func):
        with helper.raises(tp.TripyException, "Function must return 1 or more Tensors"):
            tp.compile(func, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

    def test_no_input_network(self):
        def accepts_nothing():
            return tp.Tensor([1])

        tp.compile(accepts_nothing, args=[])

    def test_multiple_return_values(self):
        compiled_func = tp.compile(
            returns_multiple_tensors,
            args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)],
        )

        a = tp.ones((2, 2), dtype=tp.float32) * 2
        b = tp.ones((2, 2), dtype=tp.float32)

        plus, minus = compiled_func(a, b)

        assert cp.array_equal(cp.from_dlpack(plus), cp.ones((2, 2), dtype=cp.float32) * 3)
        assert cp.array_equal(cp.from_dlpack(minus), cp.ones((2, 2), dtype=cp.float32))

    def test_incorrect_dtype_rejected(self):
        a = tp.ones((2, 2), dtype=tp.int32)

        with helper.raises(tp.TripyException, "Unexpected tensor data type.", has_stack_info_for=[a]):
            compiled_add = tp.compile(
                add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
            )
            compiled_add(a, a)

    def test_incorrect_shape_rejected(self):
        a = tp.ones((1, 2), dtype=tp.float32)

        with helper.raises(tp.TripyException, "Unexpected tensor shape.", has_stack_info_for=[a]):
            compiled_add = tp.compile(
                add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
            )
            compiled_add(a, a)

    @pytest.mark.skip("TODO (#155): Re-enable once we no longer implicitly copy inputs to device")
    def test_incorrect_device_rejected(self):
        compiled_add = tp.compile(
            add, args=[tp.InputInfo((2, 2), dtype=tp.float32), tp.InputInfo((2, 2), dtype=tp.float32)]
        )
        a = tp.copy(tp.ones((2, 2), dtype=tp.float32), device=tp.device("cpu"))

        with helper.raises(tp.TripyException):
            compiled_add(a, a)

    # TODO (#244): Add multi-profile test
    def test_dynamic_shapes(self):
        compiled_add = tp.compile(
            add, args=[tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32), tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32)]
        )

        out = compiled_add(tp.ones((2, 1), dtype=tp.float32), tp.ones((2, 1), dtype=tp.float32))
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 1), dtype=cp.float32) * 2)

        out = compiled_add(tp.ones((3, 1), dtype=tp.float32), tp.ones((3, 1), dtype=tp.float32))
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((3, 1), dtype=cp.float32) * 2)

    # if we specify dynamic shapes in compilation, they should not be fixed afterwards
    def test_dynamic_shapes_not_fixed(self):
        def func(inp):
            s = inp.shape[0] + inp.shape[1] + inp.shape[2]
            return tp.ones([s], dtype=tp.float32)

        compiled_ones = tp.compile(func, args=[tp.InputInfo(((1, 2, 5), (1, 2, 5), (1, 2, 5)), dtype=tp.float32)])

        for shape in ((1, 1, 1), (3, 3, 3), (2, 4, 5), (5, 2, 1)):
            inp = tp.ones(shape, dtype=tp.float32)
            out = compiled_ones(inp)
            assert out.shape == [sum(shape)]

    def test_error_if_evaling_input_during_compile(self):
        def func(a):
            print(a)
            return a + 1

        with helper.raises(tp.TripyException, match="Cannot evaluate a tensor while compiling."):
            tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.float32)])

    def test_error_if_evaling_intermediate_tensor_during_compile(self):
        def func(a):
            b = a + 1
            print(b)
            return b

        with helper.raises(tp.TripyException, match="Cannot evaluate a tensor while compiling."):
            tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.float32)])

    def test_error_if_evaling_in_nested_func_during_compile(self):
        def add(a, b):
            c = a + b
            print(c)
            return c

        def func(a):
            return add(a, 1)

        with helper.raises(tp.TripyException, match="Cannot evaluate a tensor while compiling."):
            tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.float32)])

    def test_allow_eval_if_tensor_unused_in_compile(self, capsys):
        # If the tensor is not actually used in the computation graph then we don't care if it's eval'd.
        def func(a):
            print(a.shape)

            c = a - int(a.shape[0])
            print(c)
            return a

        tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.int32)])
        out, _ = capsys.readouterr()
        print(f"\n{out}")

        # Ensure that a warning is printed for each evaluation (2 prints + int).
        assert out.count("Tensor was evaluated while compiling here:") == 3

    def test_allow_eval_for_non_input_to_compile(self):
        # We should allow non-inputs to be evaluated.
        const = tp.ones((2, 3), dtype=tp.float32)
        const.eval()

        def func(a):
            return a + const

        tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.float32)])


# TODO (#256): Remove these tests and replace with exhaustive integration testing
class TestCompiledOps:
    def test_cast(self):
        compiled_cast = tp.compile(tp.cast, args=[tp.InputInfo((2, 2), dtype=tp.float32)], kwargs=dict(dtype=tp.int32))

        a = tp.ones((2, 2), dtype=tp.float32)
        out = compiled_cast(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.int32))

    def test_linear(self):
        linear = tp.Linear(2, 3)

        compiled_linear = tp.compile(linear, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        a = tp.ones((2, 2), dtype=tp.float32)

        out = compiled_linear(a)

        assert tp.equal(out, linear(a))
