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
import pytest
from nvtripy.frontend.constraints import And, Equal, OneOf, Or, doc_str
from nvtripy.frontend.constraints.fetcher import GetDataType, GetInput


class TestDocStr:
    @pytest.mark.parametrize(
        "obj, expected",
        [
            (tp.float32, ":class:`float32`"),
            (None, "``None``"),
        ],
    )
    def test_basic_types(self, obj, expected):
        assert doc_str(obj) == expected

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
