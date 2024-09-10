#
# SPDX-FileCopyrightText: Copyright (c) 2024-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Tests that ensure that all tests do not have any randomness calls
"""

import glob
import os

import pytest
from tests import helper


class TestImports:
    INVALID_TEXT = ["random", "randn"]

    @pytest.mark.parametrize(
        "file_path",
        [file_path for file_path in glob.iglob(os.path.join(helper.ROOT_DIR, "tests", "**", "*.py"), recursive=True)],
    )
    def test_no_invalid_imports(self, file_path):
        if file_path.endswith("test_test.py"):
            with open(file_path, "r") as file:
                content = file.read()
                for invalid_text in self.INVALID_TEXT:
                    assert invalid_text not in content, f"Found '{invalid_text}' in {file_path}"
