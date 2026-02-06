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
from nvtripy.frontend.constraints import And, Equal, GetDataType, GetInput, OneOf


class TestConstraints:
    def test_find_exact_match(self):
        constraint = Equal(GetInput("a"), GetInput("b"))
        pattern = Equal(GetInput("a"), GetInput("b"))
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is constraint

    def test_find_no_match(self):
        constraint = Equal(GetInput("a"), GetInput("b"))
        pattern = OneOf(GetInput("a"), [1, 2, 3])
        assert len(constraint.find(pattern)) == 0

    def test_find_in_nested_and(self):
        inner_constraint = Equal(GetInput("a"), GetInput("b"))
        constraint = And(inner_constraint, OneOf(GetInput("c"), [1, 2, 3]))
        pattern = Equal(GetInput, GetInput)
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is inner_constraint

    def test_find_multiple_matches(self):
        equal1 = Equal(GetInput("a"), GetInput("b"))
        equal2 = Equal(GetInput("c"), GetInput("d"))
        constraint = And(equal1, equal2, OneOf(GetInput("e"), [1, 2, 3]))
        matches = constraint.find(Equal(GetInput, GetInput))
        assert len(matches) == 2 and matches[0] is equal1 and matches[1] is equal2

    def test_find_with_dtype_pattern(self):
        constraint = Equal(GetDataType(GetInput("tensor1")), GetDataType(GetInput("tensor2")))
        pattern = Equal(GetDataType(GetInput), GetDataType(GetInput))
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is constraint

    def test_find_deeply_nested_matches(self):
        equal1 = Equal(GetInput("a"), GetInput("b"))
        equal2 = Equal(GetInput("d"), GetInput("e"))
        constraint = And(
            And(equal1, OneOf(GetInput("c"), [1, 2, 3])),
            And(equal2, OneOf(GetInput("f"), [4, 5, 6])),
        )
        matches = constraint.find(Equal(GetInput, GetInput))
        assert len(matches) == 2 and matches[0] is equal1 and matches[1] is equal2

    def test_find_with_specific_names(self):
        match_constraint = Equal(GetInput("a"), GetInput("b"))
        constraint = And(match_constraint, Equal(GetInput("c"), GetInput("d")))
        matches = constraint.find(Equal(GetInput("a"), GetInput("b")))
        assert len(matches) == 1 and matches[0] is match_constraint

    def test_find_with_multiple_children(self):
        equal1 = Equal(GetInput("a"), GetInput("b"))
        equal2 = Equal(GetInput("c"), GetInput("d"))
        oneof1 = OneOf(GetInput("e"), [1, 2, 3])
        equal3 = Equal(GetInput("f"), GetInput("g"))
        constraint = And(equal1, equal2, oneof1, equal3)
        matches = constraint.find(Equal(GetInput, GetInput))
        assert len(matches) == 3
        assert equal1 in matches
        assert equal2 in matches
        assert equal3 in matches

    def test_find_and_constraint(self):
        and1 = And(Equal(GetInput("a"), GetInput("b")), OneOf(GetInput("c"), [1, 2, 3]))
        and2 = And(Equal(GetInput("d"), GetInput("e")), OneOf(GetInput("f"), [4, 5, 6]))
        constraint = And(and1, and2)
        matches = constraint.find(And(Equal, OneOf))
        assert len(matches) == 2
        assert and1 in matches
        assert and2 in matches

    def test_find_with_none_wildcard_second_arg(self):
        constraint = Equal(GetInput("a"), GetInput("b"))
        pattern = Equal(GetInput("a"), None)
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is constraint

    def test_find_with_none_wildcard_first_arg(self):
        constraint = Equal(GetInput("a"), GetInput("b"))
        pattern = Equal(None, GetInput("b"))
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is constraint

    def test_find_with_none_wildcard_in_nested(self):
        equal1 = Equal(GetDataType(GetInput("a")), GetDataType(GetInput("b")))
        equal2 = Equal(GetInput("c"), GetInput("d"))
        constraint = And(equal1, equal2)
        pattern = Equal(GetDataType(GetInput), None)
        matches = constraint.find(pattern)
        assert len(matches) == 1 and matches[0] is equal1

    def test_find_with_none_wildcard_matches_different_types(self):
        equal = Equal(GetInput("a"), GetInput("b"))
        oneof = OneOf(GetInput("c"), [1, 2, 3])
        constraint = And(equal, oneof)
        pattern = None
        matches = constraint.find(pattern)
        assert len(matches) == 6
        assert constraint in matches

    def test_info_method(self):
        constraint = Equal(GetInput("a"), GetInput("b"))
        assert constraint._info is None

        result = constraint.info("This checks that a equals b")

        assert constraint._info == "This checks that a equals b"
        assert result is constraint  # Test method chaining
