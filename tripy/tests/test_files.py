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
import pytest
from tests import helper


# Checks all Python and markdown files to ensure there are no non-ASCII characters
@pytest.mark.parametrize("file", helper.MARKDOWN_FILES + helper.PYTHON_FILES)
def test_no_non_ascii_characters(file):
    with open(file, "rb") as f:
        contents = f.read()

    try:
        contents.decode("ascii")
    except UnicodeDecodeError as err:
        str_contents = contents.decode("utf-8")

        non_ascii = str_contents[err.start : err.end]

        line_num = [line_num for line_num, line in enumerate(str_contents.splitlines()) if non_ascii in line][0] + 1

        assert False, f"Detected non-ASCII character(s) on line {line_num}: {non_ascii}"
