#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvtripy.common.exception import TripyException
from nvtripy.frontend.constraints import Equal, GetDataType, GetInput, GetReturn, NotEqual, doc_str
from tests import helper


class TestFetcher:
    def test_eq_operator_returns_equal(self):
        fetcher1 = GetInput("param1")
        fetcher2 = GetInput("param2")
        constraint = fetcher1 == fetcher2
        assert isinstance(constraint, Equal)
        assert constraint.fetcher == fetcher1
        assert constraint.fetcher_or_value == fetcher2

    def test_ne_operator_returns_not_equal(self):
        fetcher1 = GetInput("param1")
        fetcher2 = GetInput("param2")
        constraint = fetcher1 != fetcher2
        assert isinstance(constraint, NotEqual)
        assert constraint.fetcher == fetcher1
        assert constraint.fetcher_or_value == fetcher2


class TestValueFetcher:
    def test_dtype_property(self):
        fetcher = GetInput("tensor")
        dtype_fetcher = fetcher.dtype
        assert isinstance(dtype_fetcher, GetDataType)
        assert dtype_fetcher.value_fetcher == fetcher


class TestGetInput:
    def test_call(self):
        fetcher = GetInput("data")
        args = [("data", 42), ("other", "hello")]
        assert fetcher(args) == 42

    def test_str(self):
        fetcher = GetInput("data")
        assert str(fetcher) == "data"

    def test_doc_str(self):
        assert doc_str(GetInput("x")) == "``x``"


class TestGetReturn:
    def test_init(self):
        fetcher = GetReturn(0)
        assert fetcher.index == 0

    def test_call(self):
        fetcher = GetReturn(0)
        returns = (42, "hello", 3.14)
        assert fetcher([], returns) == 42

        fetcher2 = GetReturn(2)
        assert fetcher2([], returns) == 3.14

    def test_str(self):
        fetcher = GetReturn(0)
        assert str(fetcher) == "return[0]"

        fetcher2 = GetReturn(2)
        assert str(fetcher2) == "return[2]"

    def test_doc_str(self):
        assert doc_str(GetReturn(0)) == "``return[0]``"


class TestGetDataType:
    def test_call(self):
        tensor = tp.ones((2, 3), dtype=tp.float32)
        fetcher = GetDataType(GetInput("input_tensor"))
        assert fetcher([("input_tensor", tensor)]) == tp.float32

    def test_call_with_sequence(self):
        tensors = [tp.ones((2, 3), dtype=tp.float32)] * 2
        fetcher = GetDataType(GetInput("input_tensors"))
        assert fetcher([("input_tensors", tensors)]) == tp.float32

    def test_call_with_mismatched_dtypes_in_sequence(self):
        tensors = [tp.ones((2, 3), dtype=tp.float32), tp.ones((2, 3), dtype=tp.int32)]
        fetcher = GetDataType(GetInput("input_tensors"))
        with helper.raises(TripyException, match="Could not determine data type"):
            fetcher([("input_tensors", tensors)])

    def test_call_with_non_tensor_argument(self):
        fetcher = GetDataType(GetInput("input_data"))
        with helper.raises(TripyException, match="Expected a tensor or data type argument"):
            fetcher([("input_data", object())])

    def test_call_with_python_scalar_int(self):
        fetcher = GetDataType(GetInput("value"))
        assert fetcher([("value", 42)]) == tp.int32

    def test_call_with_python_scalar_float(self):
        fetcher = GetDataType(GetInput("value"))
        assert fetcher([("value", 1.0)]) == tp.float32

    def test_call_with_python_scalar_bool(self):
        fetcher = GetDataType(GetInput("value"))
        assert fetcher([("value", True)]) == tp.bool

    def test_call_with_nested_sequence_error(self):
        fetcher = GetDataType(GetInput("input_data"))
        with helper.raises(TripyException, match="Could not determine data type"):
            fetcher([("input_data", [tp.ones((2, 3), dtype=tp.float32), [42]])])

    def test_doc_str(self):
        assert doc_str(GetDataType(GetInput("x"))) == "``x.dtype``"
        assert doc_str(GetDataType(GetReturn(0))) == "``return[0].dtype``"
