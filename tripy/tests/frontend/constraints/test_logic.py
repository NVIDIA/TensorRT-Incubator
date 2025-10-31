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
from nvtripy.frontend.constraints import And, Equal, GetInput, NotEqual, NotOneOf, OneOf, Or


class TestLogic:
    def test_operator_and_basic(self):
        constraint1 = OneOf(GetInput("param1"), [1, 2, 3])
        constraint2 = OneOf(GetInput("param2"), ["a", "b", "c"])
        combined = constraint1 & constraint2
        assert isinstance(combined, And)
        assert combined([("param1", 2), ("param2", "b")])

    def test_operator_and_chaining(self):
        constraint1 = OneOf(GetInput("param1"), [1, 2, 3])
        constraint2 = OneOf(GetInput("param2"), ["a", "b", "c"])
        constraint3 = OneOf(GetInput("param3"), [True, False])
        combined = constraint1 & constraint2 & constraint3
        assert isinstance(combined, And)
        assert len(combined.constraints) == 3
        assert combined([("param1", 2), ("param2", "b"), ("param3", True)])

    def test_operator_or_basic(self):
        constraint1 = OneOf(GetInput("param1"), [1, 2, 3])
        constraint2 = OneOf(GetInput("param2"), ["a", "b", "c"])
        combined = constraint1 | constraint2
        assert isinstance(combined, Or)
        assert combined([("param1", 5), ("param2", "b")])

    def test_operator_or_chaining(self):
        constraint1 = OneOf(GetInput("param1"), [1, 2, 3])
        constraint2 = OneOf(GetInput("param2"), ["a", "b", "c"])
        constraint3 = OneOf(GetInput("param3"), [True, False])
        combined = constraint1 | constraint2 | constraint3
        assert isinstance(combined, Or)
        assert len(combined.constraints) == 3
        assert combined([("param1", 5), ("param2", "z"), ("param3", True)])

    def test_operator_not_basic(self):
        constraint = OneOf(GetInput("param"), [1, 2, 3])
        negated = ~constraint
        assert isinstance(negated, NotOneOf)
        assert negated([("param", 5)])
        assert not negated([("param", 2)])


class TestOneOf:
    def test_call(self):
        constraint = OneOf(GetInput("param"), [1, 2, 3])
        assert constraint([("param", 2)])
        result = constraint([("param", 5)])
        assert not result
        assert "'param' to be one of [1, 2, 3] (but it was '5')" in result.error_details

    def test_str(self):
        assert str(OneOf(GetInput("param"), [1, 2, 3])) == "param is one of [1, 2, 3]"

    def test_inverse(self):
        constraint = OneOf(GetInput("param"), [1, 2, 3])
        inverse = constraint.inverse()
        assert isinstance(inverse, NotOneOf)
        assert inverse([("param", 5)])
        assert not inverse([("param", 2)])


class TestNotOneOf:
    def test_call(self):
        constraint = NotOneOf(GetInput("param"), [1, 2, 3])
        assert constraint([("param", 5)])
        result = constraint([("param", 2)])
        assert not result
        assert "'param' to not be one of [1, 2, 3] (but it was '2')" in result.error_details

    def test_str(self):
        assert str(NotOneOf(GetInput("param"), [1, 2, 3])) == "param is not one of [1, 2, 3]"

    def test_inverse(self):
        constraint = NotOneOf(GetInput("param"), [1, 2, 3])
        inverse = constraint.inverse()
        assert isinstance(inverse, OneOf)
        assert inverse([("param", 2)])
        assert not inverse([("param", 5)])


class TestAnd:
    def test_call_all_pass(self):
        and_constraint = And(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        assert and_constraint([("param1", 2), ("param2", "b")])

    def test_call_one_fails(self):
        and_constraint = And(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        result = and_constraint([("param1", 5), ("param2", "b")])
        assert not result
        assert "'param1' to be one of [1, 2, 3] (but it was '5')" in result.error_details

    def test_call_all_fail(self):
        and_constraint = And(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        result = and_constraint([("param1", 5), ("param2", "z")])
        assert not result
        assert (
            "".join(result.error_details)
            == "'param1' to be one of [1, 2, 3] (but it was '5') and 'param2' to be one of ['a', 'b', 'c'] (but it was 'z')"
        )

    def test_str(self):
        and_constraint = And(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b"]))
        assert str(and_constraint) == "(param1 is one of [1, 2, 3] and param2 is one of ['a', 'b'])"

    def test_inverse(self):
        # De Morgan's law: not (A and B) = (not A) or (not B)
        and_constraint = And(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b"]))
        inverse = and_constraint.inverse()
        assert isinstance(inverse, Or)
        assert str(inverse) == "(param1 is not one of [1, 2, 3] or param2 is not one of ['a', 'b'])"


class TestOr:
    def test_call_first_passes(self):
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        assert or_constraint([("param1", 2), ("param2", "z")])

    def test_call_second_passes(self):
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        assert or_constraint([("param1", 5), ("param2", "b")])

    def test_call_all_pass(self):
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        assert or_constraint([("param1", 2), ("param2", "b")])

    def test_call_all_fail(self):
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b", "c"]))
        result = or_constraint([("param1", 5), ("param2", "z")])
        assert not result
        assert (
            "".join(result.error_details)
            == "'param1' to be one of [1, 2, 3] (but it was '5') or 'param2' to be one of ['a', 'b', 'c'] (but it was 'z')"
        )

    def test_str(self):
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b"]))
        assert str(or_constraint) == "(param1 is one of [1, 2, 3] or param2 is one of ['a', 'b'])"

    def test_call_multiple_constraints(self):
        or_constraint = Or(
            OneOf(GetInput("param1"), [1, 2, 3]),
            OneOf(GetInput("param2"), ["a", "b", "c"]),
            OneOf(GetInput("param3"), [True, False]),
        )
        assert or_constraint([("param1", 5), ("param2", "z"), ("param3", True)])
        assert not or_constraint([("param1", 5), ("param2", "z"), ("param3", None)])

    def test_inverse(self):
        # De Morgan's law: not (A or B) = (not A) and (not B)
        or_constraint = Or(OneOf(GetInput("param1"), [1, 2, 3]), OneOf(GetInput("param2"), ["a", "b"]))
        inverse = or_constraint.inverse()
        assert isinstance(inverse, And)
        assert str(inverse) == "(param1 is not one of [1, 2, 3] and param2 is not one of ['a', 'b'])"


class TestEqual:
    def test_call(self):
        constraint = Equal(GetInput("param1"), GetInput("param2"))
        assert constraint([("param1", 5), ("param2", 5)])
        result = constraint([("param1", 5), ("param2", 10)])
        assert not result
        assert "'param1' to be equal to 'param2' (but it was '5')" in result.error_details

    def test_str(self):
        assert str(Equal(GetInput("param1"), GetInput("param2"))) == "param1 == param2"
        assert str(Equal(GetInput("param1"), 5)) == "param1 == 5"

    def test_operator_on_fetcher(self):
        constraint = GetInput("param1") == GetInput("param2")
        assert isinstance(constraint, Equal)

    def test_inverse(self):
        constraint = Equal(GetInput("param1"), 5)
        inverse = constraint.inverse()
        assert isinstance(inverse, NotEqual)
        assert inverse([("param1", 10)])
        assert not inverse([("param1", 5)])


class TestNotEqual:
    def test_call(self):
        constraint = NotEqual(GetInput("param1"), GetInput("param2"))
        assert constraint([("param1", 5), ("param2", 10)])
        result = constraint([("param1", 5), ("param2", 5)])
        assert not result
        assert "'param1' to be not equal to 'param2' (but it was '5')" in result.error_details

    def test_str(self):
        assert str(NotEqual(GetInput("param1"), GetInput("param2"))) == "param1 != param2"
        assert str(NotEqual(GetInput("param1"), 5)) == "param1 != 5"

    def test_operator_on_fetcher(self):
        constraint = GetInput("param1") != GetInput("param2")
        assert isinstance(constraint, NotEqual)

    def test_inverse(self):
        constraint = NotEqual(GetInput("param1"), 5)
        inverse = constraint.inverse()
        assert isinstance(inverse, Equal)
        assert inverse([("param1", 5)])
        assert not inverse([("param1", 10)])
