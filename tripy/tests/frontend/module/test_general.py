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

import dataclasses
import inspect

import pytest

from tests import helper
from nvtripy.frontend.module import Module

MODULE_TYPES = sorted(
    {
        obj
        for obj in helper.discover_tripy_objects()
        if inspect.isclass(obj) and issubclass(obj, Module) and obj is not Module
    },
    key=lambda cls: cls.__name__,
)


@pytest.mark.parametrize("ModuleType", MODULE_TYPES)
class TestModules:
    def test_is_dataclass(self, ModuleType):
        assert dataclasses.is_dataclass(
            ModuleType
        ), f"Modules must be data classes so that we can ensure attributes have type annotations and documentation"
