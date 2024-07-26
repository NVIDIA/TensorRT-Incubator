
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

"""
Tests that ensure internal documentation (for developers) is correct.
"""

from typing import Dict, List

import pytest

from tests import helper
import tempfile

# Maps out code blocks for each README. The first element in the tuple is the README path.
DOC_CODE_SNIPPETS: Dict[str, List[helper.ReadmeCodeBlock]] = []


DOC_CODE_SNIPPETS = {
    readme: [
        code_block
        for code_block in helper.consolidate_code_blocks_from_readme(readme)
        if "py" in code_block.lang and not code_block.has_marker("ignore")
    ]
    for readme in helper.MARKDOWN_FILES
}

DOC_CODE_SNIPPETS = {readme: code_blocks for readme, code_blocks in DOC_CODE_SNIPPETS.items() if code_blocks}


@pytest.mark.parametrize(
    "code_blocks",
    DOC_CODE_SNIPPETS.values(),
    ids=DOC_CODE_SNIPPETS.keys(),
)
def test_python_code_snippets(code_blocks):
    all_pytest = all(block.has_marker("pytest") for block in code_blocks)
    assert (
        not any(block.has_marker("pytest") for block in code_blocks) or all_pytest
    ), f"This test does not currently support mixing blocks meant to be run with PyTest with blocks meant to be run by themselves!"

    # We concatenate all the code together because most documentation includes code
    # that is continued from previous code blocks.
    # TODO: We can instead run the blocks individually and propagate the evaluated local variables like `generate_rsts.py` does.
    code = "\n\n".join(map(str, code_blocks))
    print(f"Checking code:\n{code}")

    if all_pytest:
        f = tempfile.NamedTemporaryFile(mode="w+", suffix=".py")
        f.write(code)
        f.flush()

        assert pytest.main([f.name, "-vv", "-s"]) == 0
    else:
        helper.exec_code(code)
