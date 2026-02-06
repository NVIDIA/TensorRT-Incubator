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

from collections import defaultdict

import nvtripy as tp
import pytest
from nvtripy import utils
from nvtripy.frontend.wrappers import constant_fields
from tests import helper


class TestMd5:
    def test_hash_same_for_same_objects(self):
        assert utils.utils.md5(0) == utils.utils.md5(0)

    def test_hash_different_for_different_objects(self):
        assert utils.utils.md5(0) != utils.utils.md5(1)

    @pytest.mark.parametrize(
        "func",
        [
            # Check devices
            lambda: tp.device("cpu"),
            lambda: tp.device("cpu:0"),
            lambda: tp.device("gpu:0"),
        ],
    )
    def test_hash_equivalence(self, func):
        obj0 = func()
        obj1 = func()
        assert utils.utils.md5(obj0) == utils.utils.md5(obj1)


def make_with_constant_field():
    @constant_fields("field")
    class WithConstField:
        def __init__(self):
            self.custom_setter_called_count = defaultdict(int)
            self.field = 0
            self.other_field = 1

        def __setattr__(self, name, value):
            if name != "custom_setter_called_count":
                self.custom_setter_called_count[name] += 1
            return super().__setattr__(name, value)

    return WithConstField()


@pytest.fixture()
def with_const_field():
    yield make_with_constant_field()


class TestConstantFields:
    def test_field_is_immuatable(self, with_const_field):
        with helper.raises(
            tp.TripyException, match="Field: 'field' of class: '[a-zA-Z<>._]+?WithConstField' is immutable"
        ):
            with_const_field.field = 1

    def test_does_not_affect_other_fields(self, with_const_field):
        with_const_field.other_field = 3

    def test_does_not_override_custom_setter(self, with_const_field):
        assert with_const_field.custom_setter_called_count["other_field"] == 1
        with_const_field.other_field = 2
        assert with_const_field.custom_setter_called_count["other_field"] == 2

    def test_is_per_instance(self):
        const0 = make_with_constant_field()
        # When we initialize the `field` value for the second instance, it should NOT fail due to
        # the first instance already having initialized the field. This could happen if the implementation
        # doesn't take the instance into account when checking if the field has been initialized.
        const1 = make_with_constant_field()


class TestUniqueNameGen:
    @pytest.mark.parametrize(
        "inputs, outputs, expected_prefix",
        [
            (None, None, "_"),
            (["a"], None, "ins_a_"),
            (None, ["b"], "outs_b_"),
            (["a"], ["b"], "ins_a_outs_b_"),
            (["a", "b"], ["c", "d"], "ins_a_b_outs_c_d_"),
        ],
    )
    def test_gen_uid(self, inputs, outputs, expected_prefix):
        for suffix in range(2):
            uid = utils.utils.UniqueNameGen.gen_uid(inputs, outputs)
            assert uid == f"{expected_prefix}{suffix}"

    def test_uniqueness(self):
        uids = [utils.utils.UniqueNameGen.gen_uid() for _ in range(100)]
        assert len(set(uids)) == 100


class TestMergeFunctionArguments:
    def test_defaults_filled_in_with_mixed_args(self):
        # Tests that defaults are filled in and positioned correctly before kwargs
        def func(a, b=2, c=3, d=4):
            pass

        all_args, omitted_args, var_arg_info = utils.utils.merge_function_arguments(func, 1, d=99)
        assert all_args == [("a", 1), ("d", 99)]
        assert omitted_args == [("b", 2), ("c", 3)]
        assert var_arg_info is None

    def test_variadic_positional_args(self):
        # Tests variadic args tracking and defaults after variadic args
        def func(a, *args, b=10):
            pass

        all_args, omitted_args, var_arg_info = utils.utils.merge_function_arguments(func, 1, 2, 3)
        assert all_args == [("a", 1), ("args", 2), ("args", 3)]
        assert omitted_args == [("b", 10)]
        assert var_arg_info == ("args", 1)

    def test_no_variadic_args_provided(self):
        # Tests that var_arg_info is None when no variadic args are provided
        def func(a, *args, b=10):
            pass

        all_args, omitted_args, var_arg_info = utils.utils.merge_function_arguments(func, 1)
        assert all_args == [("a", 1)]
        assert omitted_args == [("b", 10)]
        assert var_arg_info is None
