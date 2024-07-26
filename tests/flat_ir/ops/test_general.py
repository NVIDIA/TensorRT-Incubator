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

import dataclasses
import inspect

import pytest

from tests import helper
from tripy.flat_ir.ops import BaseFlatIROp

OP_TYPES = {obj for obj in helper.discover_tripy_objects() if inspect.isclass(obj) and issubclass(obj, BaseFlatIROp)}


@pytest.mark.parametrize("OpType", OP_TYPES)
class TestFlatIROps:
    def test_is_dataclass(self, OpType):
        assert dataclasses.is_dataclass(
            OpType
        ), f"FlatIR ops must be data classes since many base implementations rely on dataclass introspection"

    def test_has_no_dataclass_repr(self, OpType):
        # If you define a custom repr, add a waiver here.
        assert (
            OpType.__repr__ is BaseFlatIROp.__repr__
        ), "Use @dataclass(repr=False) to avoid extremely verbose __repr__ implementations"
