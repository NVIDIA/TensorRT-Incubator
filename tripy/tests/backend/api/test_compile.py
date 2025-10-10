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
from typing import Tuple

import cupy as cp
import nvtripy as tp
import pytest
from tests import helper
from tests.backend.api.conftest import *


class TestCompile:
    # TODO (#246): Verify that it's actually compiling somehow here and below.
    # Need to return something programatically queriable from compile to do this.
    def test_function(self):
        compiled_relu = tp.compile(tp.relu, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        inp = (tp.iota((2, 2), dtype=tp.float32) - 1).eval()
        out = compiled_relu(inp)

        assert tp.equal(out, tp.relu(inp))

    def test_module(self):
        layernorm = tp.LayerNorm(2)
        layernorm.weight = tp.ones(layernorm.weight.shape)
        layernorm.bias = tp.ones(layernorm.bias.shape)

        compiled_layernorm = tp.compile(layernorm, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        inp = (tp.iota((2, 2), dtype=tp.float32) - 1).eval()
        out = compiled_layernorm(inp)

        assert tp.equal(out, layernorm(inp))

    def test_compile_preserves_sequence_for_single_output(self):
        def func(a):
            return [tp.relu(a)]

        compiled_func = tp.compile(func, args=[tp.InputInfo((2, 2), dtype=tp.float32)])

        assert inspect.signature(compiled_func).return_annotation == Tuple[tp.Tensor]

        inp = (tp.iota((2, 2), dtype=tp.float32) - 1).eval()
        [out] = compiled_func(inp)

        assert tp.equal(out, tp.relu(inp))

    def test_can_compile_using_shape_of_tensor(self):
        # Since InputInfo allows `DimensionSize`s, we should be able to use the shape of a tensor as
        # the shape of the InputInfo.
        inp = (tp.iota((2, 2), dtype=tp.float32) - 1).eval()
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

        a = (tp.ones((2, 2), dtype=tp.float32) * 2).eval()
        b = tp.ones((2, 2), dtype=tp.float32).eval()

        # Compiled function should still take arguments in (a, b) order.
        out = compiled_sub(a, b)
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32))

    @pytest.mark.parametrize("b", [2, tp.ones((2, 2), dtype=tp.float32) * 2])
    def test_constants_baked(self, b):
        # Any non-InputInfo argument to compile is baked into the compiled function.
        compiled_add = tp.compile(add, args=[tp.InputInfo((2, 2), dtype=tp.float32), b])

        a = tp.zeros((2, 2), dtype=tp.float32).eval()

        out = compiled_add(a)

        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 2), dtype=cp.float32) * 2)

    @pytest.mark.parametrize("num_args", [1, 2, 3])
    def test_variadic_positional_args(self, num_args):
        func = tp.compile(
            variadic_positional,
            args=[tp.InputInfo((2,), dtype=tp.float32) for _ in range(num_args)],
        )

        inputs = [tp.ones((2,)).eval() for _ in range(num_args)]
        assert tp.equal(func(*inputs), tp.ones((2,)) * num_args)

    @pytest.mark.parametrize("num_args", [1, 2, 3])
    def test_variadic_keyword_args(self, num_args):
        func = tp.compile(
            variadic_keyword,
            kwargs={f"arg{index}": tp.InputInfo((2,), dtype=tp.float32) for index in range(num_args)},
        )

        inputs = {f"arg{index}": tp.ones((2,)).eval() for index in range(num_args)}
        assert tp.equal(func(**inputs), tp.ones((2,)) * num_args)

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

        a = (tp.ones((2, 2), dtype=tp.float32) * 2).eval()
        b = tp.ones((2, 2), dtype=tp.float32).eval()

        plus, minus = compiled_func(a, b)

        assert cp.array_equal(cp.from_dlpack(plus), cp.ones((2, 2), dtype=cp.float32) * 3)
        assert cp.array_equal(cp.from_dlpack(minus), cp.ones((2, 2), dtype=cp.float32))

    # TODO (#244): Add multi-profile test
    def test_dynamic_shapes(self):
        compiled_add = tp.compile(
            add, args=[tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32), tp.InputInfo(((1, 2, 3), 1), dtype=tp.float32)]
        )

        out = compiled_add(tp.ones((2, 1), dtype=tp.float32).eval(), tp.ones((2, 1), dtype=tp.float32).eval())
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((2, 1), dtype=cp.float32) * 2)

        out = compiled_add(tp.ones((3, 1), dtype=tp.float32).eval(), tp.ones((3, 1), dtype=tp.float32).eval())
        assert cp.array_equal(cp.from_dlpack(out), cp.ones((3, 1), dtype=cp.float32) * 2)

    # if we specify dynamic shapes in compilation, they should not be fixed afterwards
    def test_dynamic_shapes_not_fixed(self):
        def func(inp):
            s = inp.shape[0] + inp.shape[1] + inp.shape[2]
            return tp.ones([s], dtype=tp.float32)

        compiled_ones = tp.compile(func, args=[tp.InputInfo(((1, 2, 5), (1, 2, 5), (1, 2, 5)), dtype=tp.float32)])

        for shape in ((1, 1, 1), (3, 3, 3), (2, 4, 5), (5, 2, 1)):
            inp = tp.ones(shape, dtype=tp.float32).eval()
            out = compiled_ones(inp)
            assert out.shape == (sum(shape),)

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

    def test_disallow_eval_for_non_input_to_compile(self):
        # Non-inputs cannot be evaluated since then they move to GPU memory,
        # which is not supported for constants.
        const = tp.ones((2, 3), dtype=tp.float32)
        const.eval()

        def func(a):
            return a + const

        with helper.raises(
            tp.TripyException, match="Tensors that are not inputs to compiled functions must reside in CPU memory."
        ):
            tp.compile(func, args=[tp.InputInfo((2, 3), dtype=tp.float32)])

    def test_dimension_input(self):
        dummy = tp.ones((3, 4))
        compiled = tp.compile(
            dynamic_reshape,
            args=[
                tp.InputInfo(shape=((2, 4, 6), 4), dtype=tp.float32),
                tp.DimensionInputInfo(value_bounds=(2, dummy.shape[1], 6)),
            ],
        )
        for reshape_dim in [2, 4, 6]:
            inp_cp = cp.arange(12, dtype=cp.float32).reshape((3, 4))
            inp = tp.Tensor(inp_cp)
            dim_inp = tp.DimensionSize(reshape_dim)
            out = compiled(inp, dim_inp)
            expected = (inp_cp + inp_cp).reshape((-1, reshape_dim))
            assert cp.array_equal(cp.from_dlpack(out), expected)

    def test_compile_dict_input_info(self):
        """Test compilation with dictionary of InputInfo objects."""

        def func(data_dict):
            return data_dict["a"] + data_dict["b"]

        dict_input = {
            "a": tp.InputInfo(shape=(2, 3), dtype=tp.float32),
            "b": tp.InputInfo(shape=(2, 3), dtype=tp.float32),
        }
        compiled_func = tp.compile(func, args=[dict_input])

        test_dict = {"a": tp.ones((2, 3), dtype=tp.float32).eval(), "b": (tp.ones((2, 3), dtype=tp.float32) * 2).eval()}
        result = compiled_func(test_dict)
        expected = test_dict["a"] + test_dict["b"]
        assert cp.array_equal(cp.from_dlpack(result), cp.from_dlpack(expected))

    def test_compile_nested_list_input_info(self):
        """Test compilation with nested list containers."""

        def func(data_list):
            return data_list[0] + data_list[1][0] + data_list[1][1]

        list_input = [
            tp.InputInfo(shape=(2, 3), dtype=tp.float32),
            [  # Nested list
                tp.InputInfo(shape=(2, 3), dtype=tp.float32),
                tp.ones((2, 3), dtype=tp.float32) * 2,  # Constant in nested list
            ],
        ]
        compiled_func = tp.compile(func, args=[list_input])

        test_list = [
            tp.ones((2, 3), dtype=tp.float32).eval(),
            [  # Nested list in test data
                (tp.ones((2, 3), dtype=tp.float32) * 3).eval(),
                tp.ones((2, 3), dtype=tp.float32) * 2,  # Should match baked constant
            ],
        ]
        result = compiled_func(test_list)
        expected = test_list[0] + test_list[1][0] + test_list[1][1]
        assert cp.array_equal(cp.from_dlpack(result), cp.from_dlpack(expected))

    def test_compile_mixed_containers_and_constants(self):
        """Test compilation with comprehensive mix: regular InputInfo, dict container, list container, and standalone constant."""

        def func(regular_input, data_dict, data_list, constant_value):
            return regular_input + data_dict["x"] + data_dict["y"] + data_list[0] + data_list[1] + constant_value

        regular_input = tp.InputInfo(shape=(2, 3), dtype=tp.float32)
        dict_input = {
            "x": tp.InputInfo(shape=(2, 3), dtype=tp.float32),
            "y": tp.zeros((2, 3), dtype=tp.float32),  # Constant in dict
        }
        list_input = [tp.InputInfo(shape=(2, 3), dtype=tp.float32), tp.ones((2, 3), dtype=tp.float32) * 3]
        constant_value = tp.ones((2, 3), dtype=tp.float32) * 5

        compiled_func = tp.compile(func, args=[regular_input, dict_input, list_input, constant_value])

        # Only InputInfo arguments should be in function signature
        test_regular = tp.ones((2, 3), dtype=tp.float32).eval()
        test_dict = {"x": (tp.ones((2, 3), dtype=tp.float32) * 2).eval(), "y": tp.zeros((2, 3), dtype=tp.float32)}
        test_list = [(tp.ones((2, 3), dtype=tp.float32) * 4).eval(), tp.ones((2, 3), dtype=tp.float32) * 3]

        result = compiled_func(test_regular, test_dict, test_list)
        expected = test_regular + test_dict["x"] + test_dict["y"] + test_list[0] + test_list[1] + constant_value
        assert cp.array_equal(cp.from_dlpack(result), cp.from_dlpack(expected))
