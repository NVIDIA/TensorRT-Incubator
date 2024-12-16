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

import sys

import pytest

from nvtripy.utils.ast import get_arg_candidate_column_offsets


class TestAst:
    @pytest.mark.parametrize(
        ("call_str", "func_name", "arg_names", "num_positional", "num_total_args", "expected"),
        [
            # Function calls
            #
            # Just the column positions of the two args
            ("f(0, 1)", "f", ["a", "b"], 2, 2, [[(2, 3)], [(5, 6)]]),
            # In actual usage (due to merge_function_arguments), we do not get argument 0 (self), so we should make
            # sure arguments 1 through 3 are indexed correctly.
            (
                "obj.method('hello', 3, var)",
                "method",
                ["self", "i", "j", "k"],
                4,
                4,
                [None, [(11, 18)], [(20, 21)], [(23, 26)]],
            ),
            # For kwargs, the column range includes the name. We also include the kwargs in the arg names
            (
                "g(1 + 2, h('hello'), bonus=3.14159, one_more='encore')",
                "g",
                ["argument_1", "argument_2", "bonus", "one_more"],
                2,
                4,
                [[(2, 7)], [(9, 19)], [(21, 34)], [(36, 53)]],
            ),
            # In the concrete call, we may know the specific number of args passed for a var args, but the source code
            # may have a starred list. (This happens with slice_helper.) Note that we repeat the name of the starred
            # variable when that is the case (see merge_function_arguments).
            #
            # For the var args, they all point to the starred list.
            # The number of positional args reflects the concrete number of positional args used, hence 4 in this case.
            (
                "func('a', *args, extra='hello')",
                "func",
                ["first", "args", "args", "args", "extra"],
                4,
                5,
                [[(5, 8)], [(10, 15)], [(10, 15)], [(10, 15)], [(17, 30)]],
            ),
            #
            # Subscripts: See slice_helper for the specifics of how this is used (it's a special case).
            #
            # For subscripts, the func name is always __getitem__. We desugar single indices into a slice like 2:3:1,
            # so [2] corresponds to 3 tensor parameters. Also, only the first arg (the value being subscripted)
            # is positional because of how slice_helper works. We also disregard the base of the subscript because its stack info does not get set by slice_helper.
            (
                "a[2]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(2, 3)], [(2, 3)]],
            ),
            # The start and stop are written out explicitly, but the step is implied by the whole range.
            (
                "a[2:3]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(4, 5)], [(2, 5)]],
            ),
            # The step is written out too
            (
                "a[2:3:-1]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(4, 5)], [(6, 8)]],
            ),
            # Stop only, with step
            (
                "a[2::]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(2, 5)], [(2, 5)]],
            ),
            # Start only, with step
            (
                "a[:5:]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 5)], [(3, 4)], [(2, 5)]],
            ),
            # Step only
            (
                "a[::-1]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 6)], [(2, 6)], [(4, 6)]],
            ),
            # Start, step implicit
            (
                "a[2:]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(2, 4)], [(2, 4)]],
            ),
            # Stop, step implicit
            (
                "a[:5]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 4)], [(3, 4)], [(2, 4)]],
            ),
            # Whole range omitted
            (
                "a[:]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 3)], [(2, 3)], [(2, 3)]],
            ),
            # For multiple slice arguments, we expand each into start, stop, and step
            (
                "a[:, :]",
                "__getitem__",
                [
                    "value",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                ],
                1,
                7,
                [None, [(2, 3)], [(2, 3)], [(2, 3)], [(5, 6)], [(5, 6)], [(5, 6)]],
            ),
            (
                "a[0, 3]",
                "__getitem__",
                [
                    "value",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                ],
                1,
                7,
                [None, [(2, 3)], [(2, 3)], [(2, 3)], [(5, 6)], [(5, 6)], [(5, 6)]],
            ),
            (
                "a[name:3:-1, 4:, :]",
                "__getitem__",
                [
                    "value",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                    "slice_params",
                ],
                1,
                10,
                [
                    None,
                    [(2, 6)],
                    [(7, 8)],
                    [(9, 11)],
                    [(13, 14)],
                    [(13, 15)],
                    [(13, 15)],
                    [(17, 18)],
                    [(17, 18)],
                    [(17, 18)],
                ],
            ),
            #
            # Binary ops: The op names will be those of the magic methods. For these, we do consider the "self" argument.
            #
            ("a + b", "__add__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a - b", "__sub__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a * b", "__mul__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a / b", "__truediv__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a // b", "__floordiv__", ["self", "other"], 2, 2, [[(0, 1)], [(5, 6)]]),
            ("a % b", "__mod__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a ** b", "__pow__", ["self", "other"], 2, 2, [[(0, 1)], [(5, 6)]]),
            ("a & b", "__and__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a | b", "__or__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a ^ b", "__xor__", ["self", "other"], 2, 2, [[(0, 1)], [(4, 5)]]),
            ("a << b", "__lshift__", ["self", "other"], 2, 2, [[(0, 1)], [(5, 6)]]),
            ("a >> b", "__rshift__", ["self", "other"], 2, 2, [[(0, 1)], [(5, 6)]]),
            #
            # No candidate cases
            #
            # None of the above AST nodes
            ("class C: pass", "my_func", ["a", "b", "c"], 3, 3, [[], [], []]),
            # Correct call but wrong name (also applies to magic methods)
            ("not_my_func(a, b, c)", "my_func", ["a", "b", "c"], 3, 3, [[], [], []]),
            ("a[0]", "not_getitem", ["a", "b", "c", "d"], 4, 4, [[], [], [], []]),
            ("a + b", "not_add", ["a", "b", "c"], 3, 3, [[], [], []]),
            #
            # Multiple candidate cases: Recursion is when we can have multiple candidates.
            #
            # Nesting is okay if names do not match
            (
                "func(a + b + c, value[1:3:-1], g(d*e + f))",
                "func",
                ["a", "b", "c"],
                3,
                3,
                [[(5, 14)], [(16, 29)], [(31, 41)]],
            ),
            #
            # Same func appears twice. We can match each arg to either one of the calls
            (
                "func(1, 2, func(2, 3, 4))",
                "func",
                ["a", "b", "c"],
                3,
                3,
                [[(5, 6), (16, 17)], [(8, 9), (19, 20)], [(11, 24), (22, 23)]],
            ),
            # Subscript as argument to a subscript.
            (
                "a[b[0]]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 6), (4, 5)], [(2, 6), (4, 5)], [(2, 6), (4, 5)]],
            ),
            # Also with slices.
            (
                "a[b[0:3:6]:3:4]",
                "__getitem__",
                ["value", "slice_params", "slice_params", "slice_params"],
                1,
                4,
                [None, [(2, 10), (4, 5)], [(11, 12), (6, 7)], [(13, 14), (8, 9)]],
            ),
            # It can happen with binops. Note that the parens are disregarded in the column range.
            (
                "(a + b) + (c + d)",
                "__add__",
                ["self", "other"],
                2,
                2,
                [[(1, 6), (1, 2), (11, 12)], [(11, 16), (5, 6), (15, 16)]],
            ),
        ],
        ids=[
            "basic_call",
            "method_call",
            "call_with_kwargs",
            "call_with_var_args",
            "basic_subscript",
            "subscript_with_start_and_stop",
            "subscript_explicit_step",
            "subscript_start_only",
            "subscript_stop_only",
            "subscript_step_only",
            "subscript_start_step_implicit",
            "subscript_stop_step_implicit",
            "subscript_range_implicit",
            "subscript_multiple_range_implicit",
            "subscript_multiple_dims",
            "subscript_multiple_slices",
            "binop_add",
            "binop_sub",
            "binop_mul",
            "binop_div",
            "binop_floordiv",
            "binop_mod",
            "binop_pow",
            "binop_and",
            "binop_or",
            "binop_xor",
            "binop_lshift",
            "binop_rshift",
            "empty_wrong_node",
            "empty_wrong_name_on_call",
            "empty_wrong_name_on_subscript",
            "empty_wrong_name_on_binop",
            "nested_but_names_do_not_match",
            "multiple_ranges_multiple_call",
            "multiple_ranges_nested_subscript",
            "multiple_ranges_nested_subscript_slices",
            "multiple_ranges_nested_binops",
        ],
    )
    def test_get_arg_candidate_column_offsets(
        self, call_str, func_name, arg_names, num_positional, num_total_args, expected
    ):
        for i in range(num_total_args):
            # for cases like method calls, where we disregard the "self" argument
            if expected[i] is None:
                continue
            assert (
                get_arg_candidate_column_offsets(call_str, i, num_positional, func_name, i >= num_positional, arg_names)
                == expected[i]
            )
