
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

import tripy as tp
from tripy.common.datatype import DATA_TYPES


class TestDataType:
    @pytest.mark.parametrize("name", DATA_TYPES.keys())
    def test_api(self, name):
        # Make sure we can access data types at the top-level, e.g. `tripy.float32`
        assert isinstance(getattr(tp, name), tp.dtype)

    @pytest.mark.parametrize("name, dtype", DATA_TYPES.items())
    def test_name(self, name, dtype):
        assert name == dtype.name

    @pytest.mark.parametrize("dtype", DATA_TYPES.values())
    def test_itemsize(self, dtype):
        EXPECTED_ITEMSIZES = {
            "float32": 4,
            "float16": 2,
            "float8": 1,
            "bfloat16": 2,
            "int4": 0.5,
            "int8": 1,
            "int32": 4,
            "int64": 8,
            "uint8": 1,
            "bool": 1,
        }
        assert dtype.itemsize == EXPECTED_ITEMSIZES[dtype.name]
