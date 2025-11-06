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

import nvtripy as tp
from nvtripy.frontend.constraints import And, Equal, NotEqual, NotOneOf, OneOf, Or, doc_str
from nvtripy.frontend.constraints.fetcher import GetDataType, GetInput, GetReturn


class TestDocStr:
    def test_basic_types(self):
        assert doc_str(tp.float32) == ":class:`float32`"
        assert doc_str(GetInput("x")) == "``x``"
        assert doc_str(GetReturn(0)) == "``return[0]``"

    def test_get_datatype(self):
        assert doc_str(GetDataType(GetInput("x"))) == "``x.dtype``"
        assert doc_str(GetDataType(GetReturn(0))) == "``return[0].dtype``"

    def test_one_of_and_not_one_of(self):
        input_x = GetInput("x")

        assert (
            doc_str(OneOf(input_x, [tp.float32, tp.float16])) == "``x`` is one of [:class:`float32`, :class:`float16`]"
        )
        assert doc_str(NotOneOf(input_x, [tp.int8, tp.int32])) == "``x`` is not one of [:class:`int8`, :class:`int32`]"

    def test_equal_and_not_equal(self):
        input_a = GetInput("a")
        input_b = GetInput("b")

        assert doc_str(Equal(input_a, input_b)) == "``a`` == ``b``"
        assert doc_str(Equal(input_a, tp.float32)) == "``a`` == :class:`float32`"
        assert doc_str(NotEqual(input_a, input_b)) == "``a`` != ``b``"

    def test_and_constraint(self):
        constraint1 = OneOf(GetInput("a"), [tp.float32])
        constraint2 = OneOf(GetInput("b"), [tp.int32])

        assert (
            doc_str(And(constraint1, constraint2))
            == "- ``a`` is one of [:class:`float32`], **and**\n- ``b`` is one of [:class:`int32`]"
        )

    def test_or_constraint(self):
        input_a = GetInput("a")
        or_constraint = Or(Equal(input_a, tp.float32), Equal(input_a, tp.float16))

        assert doc_str(or_constraint) == "(``a`` == :class:`float32` *or* ``a`` == :class:`float16`)"

    def test_nested_constraints(self):
        input_a = GetInput("a")
        input_b = GetInput("b")

        or_part = Or(Equal(input_a, tp.float32), Equal(input_a, tp.float16))
        and_constraint = And(or_part, OneOf(input_b, [tp.int32]))

        assert (
            doc_str(and_constraint)
            == "- (``a`` == :class:`float32` *or* ``a`` == :class:`float16`), **and**\n- ``b`` is one of [:class:`int32`]"
        )

    def test_complex_real_world_constraint(self):
        input_a = GetInput("input")
        input_b = GetInput("other")
        dtype_a = GetDataType(input_a)
        dtype_b = GetDataType(input_b)

        and_constraint = And(Equal(dtype_a, dtype_b), OneOf(dtype_a, [tp.float32, tp.float16]))

        assert (
            doc_str(and_constraint)
            == "- ``input.dtype`` == ``other.dtype``, **and**\n- ``input.dtype`` is one of [:class:`float32`, :class:`float16`]"
        )
