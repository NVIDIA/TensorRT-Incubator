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

from nvtripy.common.datatype import DATA_TYPES
from nvtripy.frontend.constraints import AlwaysTrue, And, GetInput, OneOf
from nvtripy.frontend.constraints.fetcher import GetDataType
from nvtripy.frontend.constraints.optimizer import optimize_constraints


class TestOptimizeConstraints:
    def test_drops_all_dtype_oneof(self):
        constraint = OneOf(GetDataType(GetInput("x")), list(DATA_TYPES.values()))
        optimized = optimize_constraints(constraint)
        assert isinstance(optimized, AlwaysTrue)

    def test_keeps_non_exhaustive_oneof(self):
        dtypes = list(DATA_TYPES.values())
        constraint = OneOf(GetDataType(GetInput("x")), dtypes[:-1])
        optimized = optimize_constraints(constraint)
        assert optimized is constraint

    def test_applies_to_nested_constraints(self):
        constraint = And(
            OneOf(GetDataType(GetInput("x")), list(DATA_TYPES.values())),
            OneOf(GetInput("y"), [1, 2, 3]),
        )
        optimized = optimize_constraints(constraint)
        assert isinstance(optimized, And)
        assert any(isinstance(child, AlwaysTrue) for child in optimized.constraints)
