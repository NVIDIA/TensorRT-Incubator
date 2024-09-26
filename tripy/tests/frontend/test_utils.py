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
import numpy as np

import tripy as tp
from tripy.frontend.utils import convert_inputs_to_tensors, convert_shape_inputs
from tests import helper

# Putting underscores at the beginning and end of the names to get around the check
# for magic methods. We would not want to see this outside of tests.


@convert_inputs_to_tensors()
def __func_test_basic__(a):
    return a


@convert_inputs_to_tensors()
def __func_test_multi_input__(a, b, c):
    return a, b, c


@convert_inputs_to_tensors(sync_arg_types=[("a", "b", "c")])
def __func_test_sync_arg_types__(a, b, c):
    return a, b, c


@convert_inputs_to_tensors()
def __func_test_variadic_positional_args__(*args):
    return args


@convert_inputs_to_tensors()
def __func_test_arg_before_variadic_positional_args__(x, *args):
    return (x,) + args


@convert_inputs_to_tensors()
def __func_test_kwarg_after_variadic_positional_args__(*args, y):
    return args + (y,)


@convert_inputs_to_tensors(unpack_argument=["xs"])
def __func_test_convert_list_input__(xs):
    return xs


@convert_inputs_to_tensors(sync_arg_types=[("xs",)], unpack_argument=["xs"])
def __func_test_sync_within_list__(xs):
    return xs


@convert_inputs_to_tensors(sync_arg_types=[("x", "ys")], unpack_argument=["ys"])
def __func_test_sync_single_type_to_list__(x, ys):
    return x, ys


@convert_inputs_to_tensors(sync_arg_types=[("xs", "y")], unpack_argument=["xs"])
def __func_test_sync_list_type_to_single__(xs, y):
    return xs, y


@convert_inputs_to_tensors(sync_arg_types=[("xs", "ys")], unpack_argument=["xs", "ys"])
def __func_test_sync_list_types__(xs, ys):
    return xs, ys


@convert_shape_inputs(["s"])
def convert_shape(s):
    return s


@convert_shape_inputs(["s"])
def ignore_not_named(s, t):
    return (s, t)


class TestConvertInputsToTensors:
    def test_args(self):
        assert isinstance(__func_test_basic__(0), tp.Tensor)

    def test_kwargs(self):
        assert isinstance(__func_test_basic__(a=0), tp.Tensor)

    def test_convert_list_into_tensor(self):
        t1 = __func_test_basic__([1, 2, 3])
        assert isinstance(t1, tp.Tensor)
        assert t1.shape == [3]

        t2 = __func_test_basic__([[1, 2], [3, 4]])
        assert t2.shape == [2, 2]

    def test_convert_list_input(self):
        xs = __func_test_convert_list_input__([1.0, 2.0, 3.0, 4.0])
        assert len(xs) == 4
        for x in xs:
            assert isinstance(x, tp.Tensor)
        assert not __func_test_convert_list_input__([])

    def test_convert_tuple_input(self):
        xs = __func_test_convert_list_input__((1.0, 2.0))
        assert isinstance(xs, tuple)
        assert len(xs) == 2
        assert isinstance(xs[0], tp.Tensor)
        assert isinstance(xs[1], tp.Tensor)

    def test_variadic_positional_args(self):
        x, y = __func_test_variadic_positional_args__(1.0, 2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_arg_before_variadic_positional_args(self):
        x, y = __func_test_arg_before_variadic_positional_args__(1.0, 2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_kwarg_after_variadic_positional_args(self):
        x, y = __func_test_kwarg_after_variadic_positional_args__(1.0, y=2.0)
        assert isinstance(x, tp.Tensor)
        assert isinstance(y, tp.Tensor)

    def test_convert_shape_basic(self):
        s = convert_shape([1, 2, 3])
        assert isinstance(s, tp.Shape)
        assert s.shape == [3]

    def test_convert_shape_already_shape(self):
        s1 = tp.Shape([1, 2, 3])
        s2 = convert_shape(s1)
        assert s1.trace_tensor == s2.trace_tensor

    def test_convert_shape_tensor(self):
        t = tp.Tensor([1, 2, 3], dtype=tp.int32)
        s2 = convert_shape(t)
        assert isinstance(s2, tp.Shape)
        assert t.trace_tensor == s2.trace_tensor

    def test_convert_mixed_type_list_to_shape(self):
        from tripy.frontend.trace.ops.concatenate import Concatenate

        t = tp.Tensor([3], dtype=tp.int32)
        s1 = tp.Shape([1, 2])
        s2 = convert_shape([1, 2, 3, s1[0], 4, 5, t[0], s1[1], 6, 7])
        assert isinstance(s2, tp.Shape)
        assert isinstance(s2.trace_tensor.producer, Concatenate)
        # ensure the concatenation is done correctly
        assert cp.from_dlpack(s2).get().tolist() == [1, 2, 3, 1, 4, 5, 3, 2, 6, 7]

    def test_negative_non_scalar_in_shape(self):
        s1 = tp.Shape([1, 2])
        with helper.raises(tp.TripyException, match="Tensor in a shape argument must be a scalar."):
            s2 = convert_shape([1, s1])

    def test_convert_empty_shape(self):
        s = convert_shape([])
        assert isinstance(s, tp.Shape)
        assert cp.from_dlpack(s).get().tolist() == []

    def test_convert_shape_unsqueeze_tensors(self):
        t1 = tp.Tensor(1, dtype=tp.int32)
        t2 = tp.Tensor(2, dtype=tp.int32)
        s = convert_shape([t1, t2])
        assert isinstance(s, tp.Shape)
        assert cp.from_dlpack(s).get().tolist() == [1, 2]

    def test_convert_only_specified_argument_to_shape(self):
        t1 = tp.Tensor([1, 2, 3], dtype=tp.int32)
        s, t2 = ignore_not_named(t1, [4, 5, 6])
        assert isinstance(s, tp.Shape)
        assert t2 == [4, 5, 6]

    # When we convert arguments to tensors, we should preserve the column range
    # of the original non-Tensor argument.
    def test_includes_column_range_for_non_tensors(self):
        tensor = __func_test_basic__(3.0)

        # Column offset of the `3.0` above.
        assert tensor.stack_info[tensor.stack_info.get_first_user_frame_index()].column_range == (37, 40)

    def test_includes_column_range_for_non_tensors_multiple_inputs(self):
        a, b, c = __func_test_multi_input__(1, 2.0, 3)

        # Column offsets of the arguments above.
        assert a.stack_info[a.stack_info.get_first_user_frame_index()].column_range == (44, 45)
        assert b.stack_info[b.stack_info.get_first_user_frame_index()].column_range == (47, 50)
        assert c.stack_info[c.stack_info.get_first_user_frame_index()].column_range == (52, 53)

    def test_includes_column_range_for_non_tensors_multiple_inputs_with_kwargs(self):
        a, b, c = __func_test_multi_input__(1, b=2.0, c=3)

        # Column offsets of the arguments above.
        assert a.stack_info[a.stack_info.get_first_user_frame_index()].column_range == (44, 45)
        assert b.stack_info[b.stack_info.get_first_user_frame_index()].column_range == (47, 52)
        assert c.stack_info[c.stack_info.get_first_user_frame_index()].column_range == (54, 57)

    def test_includes_column_range_for_non_tensors_for_magic_methods(self):
        c = tp.ones((2, 3)) + 3

        stack_info = c.trace_tensor.producer.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (30, 31)

    def test_includes_column_range_for_non_tensors_for_magic_methods_with_kwargs(self):
        c = tp.ones((2, 3)).__add__(other=3)

        stack_info = c.trace_tensor.producer.inputs[1].stack_info

        # Column offset of the `3` above.
        assert stack_info[stack_info.get_first_user_frame_index()].column_range == (36, 43)

    def test_includes_column_range_for_list_elements(self):
        xs = __func_test_convert_list_input__([1.0, 2.0])
        assert xs[0].stack_info[xs[0].stack_info.get_first_user_frame_index()].column_range == (47, 50)
        assert xs[1].stack_info[xs[1].stack_info.get_first_user_frame_index()].column_range == (52, 55)

    def test_includes_column_range_for_tuple_elements(self):
        xs = __func_test_convert_list_input__((1.0, 2.0))
        assert xs[0].stack_info[xs[0].stack_info.get_first_user_frame_index()].column_range == (47, 50)
        assert xs[1].stack_info[xs[1].stack_info.get_first_user_frame_index()].column_range == (52, 55)

    def test_sync_arg_type_includes_non_tensor_column_range(self):
        x, y, z = __func_test_sync_arg_types__(tp.Tensor(3.0, dtype=tp.float16), 3, 4.0)

        assert y.dtype == tp.float16
        assert z.dtype == tp.float16
        assert y.stack_info[y.stack_info.get_first_user_frame_index()].column_range == (81, 82)
        assert z.stack_info[z.stack_info.get_first_user_frame_index()].column_range == (84, 87)

    def test_sync_arg_type_includes_non_tensor_column_range_with_kwargs(self):
        x, y, z = __func_test_sync_arg_types__(tp.Tensor(3.0, dtype=tp.float16), b=3, c=4.0)

        assert y.dtype == tp.float16
        assert z.dtype == tp.float16
        assert y.stack_info[y.stack_info.get_first_user_frame_index()].column_range == (81, 84)
        assert z.stack_info[z.stack_info.get_first_user_frame_index()].column_range == (86, 91)

    def test_sync_arg_type_not_applied_to_tensors(self):
        x, y, z = __func_test_sync_arg_types__(
            tp.Tensor(3.0),
            tp.Tensor(3, dtype=tp.int32),
            tp.Tensor(4, dtype=tp.float16),
        )

        assert x.dtype == tp.float32
        assert y.dtype == tp.int32
        assert z.dtype == tp.float16

    def test_sync_arg_type_within_list(self):
        xs = __func_test_sync_within_list__([1.0, tp.Tensor(3, dtype=tp.float16), 5])

        assert xs[0].dtype == tp.float16
        assert xs[1].dtype == tp.float16
        assert xs[2].dtype == tp.float16

    def test_sync_single_arg_type_to_list(self):
        _, ys = __func_test_sync_single_type_to_list__(tp.Tensor(5, dtype=tp.int32), [2.0, 3.0, 4.0])

        assert ys[0].dtype == tp.int32
        assert ys[1].dtype == tp.int32
        assert ys[2].dtype == tp.int32

    def test_sync_list_arg_type_to_single_arg(self):
        xs, y = __func_test_sync_list_type_to_single__([1.0, tp.Tensor(5, dtype=tp.int32), 4.0], 1.0)

        assert xs[0].dtype == tp.int32
        assert xs[2].dtype == tp.int32
        assert y.dtype == tp.int32

    def test_sync_list_arg_types(self):
        xs, ys = __func_test_sync_list_types__([1.0, 2.0, 3.0], [3, 4, tp.Tensor(6, dtype=tp.int32)])

        for x in xs:
            assert x.dtype == tp.int32
        for y in ys:
            assert y.dtype == tp.int32

    def test_sync_arg_type_list_not_applied_to_tensors(self):
        xs = __func_test_sync_within_list__(
            [tp.Tensor(1.0, dtype=tp.int32), tp.Tensor(3, dtype=tp.float16), tp.Tensor(5, dtype=tp.float32)]
        )

        assert xs[0].dtype == tp.int32
        assert xs[1].dtype == tp.float16
        assert xs[2].dtype == tp.float32

    def test_sync_arg_type_invalid(self):
        with helper.raises(
            tp.TripyException,
            match=r"At least one of the arguments: \('a', 'b', 'c'\) must be a \`tripy.Tensor\`.",
        ):
            x, y, z = __func_test_sync_arg_types__(3.0, 3, 4)

    def test_seq_arg_invalid(self):
        with helper.raises(
            tp.TripyException,
            match=r"Encountered non-number of type str in sequence: hello",
        ):
            _ = __func_test_basic__([1, 2, "hello"])

    def test_nested_seq_inconsistent_len(self):
        with helper.raises(
            tp.TripyException,
            match=r"Expected a sequence of length 3 but got length 4: \[7, 8, 9, 10\]",
        ):
            _ = __func_test_basic__([[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]])

    def test_nested_seq_inconsistent_types(self):
        with helper.raises(
            tp.TripyException,
            match=r"Expected a sequence but got str: hi",
        ):
            _ = __func_test_basic__([[1, 2, 3], [4, 5, 6], "hi"])

    def test_invalid_argument_type_not_converted(self):
        a = np.array([1, 2, 3])
        b = __func_test_basic__(np.array([1, 2, 3]))
        assert (a == b).all()
