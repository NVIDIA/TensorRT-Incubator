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

"""
Tests that ensure internal documentation (for developers) is correct.
"""

from typing import Dict, List

import pytest

from tests import helper
import tempfile

# Maps out code blocks for each README. The first element in the tuple is the README path.
ALL_DOC_CODE_BLOCKS: Dict[str, List[helper.ReadmeCodeBlock]] = []


ALL_DOC_CODE_BLOCKS = {
    readme: [
        code_block
        for code_block in helper.consolidate_code_blocks_from_readme(readme)
        if "py" in code_block.lang and not code_block.has_marker("test: ignore")
    ]
    for readme in helper.MARKDOWN_FILES
}


# Guides may use inline pytest tests or regular Python code snippets.
INLINE_PYTESTS = {}

for readme, code_blocks in ALL_DOC_CODE_BLOCKS.items():
    if not code_blocks:
        continue

    if all(block.has_marker("test: use_pytest") for block in code_blocks):
        INLINE_PYTESTS[readme] = code_blocks
    else:
        assert not any(
            block.has_marker("test: use_pytest") for block in code_blocks
        ), "Guides must not mix Pytest code blocks with non-Pytest code blocks"


@pytest.mark.parametrize(
    "code_blocks",
    INLINE_PYTESTS.values(),
    ids=INLINE_PYTESTS.keys(),
)
def test_inline_pytest(code_blocks):
    code = "\n\n".join(map(str, code_blocks))
    f = tempfile.NamedTemporaryFile(mode="w+", suffix=".py")
    f.write(code)
    f.flush()
    assert pytest.main([f.name, "-vv", "-s"]) == 0
