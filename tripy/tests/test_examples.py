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
Tests to ensure that examples specify dependencies, run correctly, and follow our standardized
example format.
"""

import glob
import os
import re
import shutil
from typing import Sequence
from textwrap import dedent

import pytest

from tests import helper

EXAMPLES_ROOT = os.path.join(helper.ROOT_DIR, "examples")


class Example:
    def __init__(self, path_components: Sequence[str], artifact_names: Sequence[str] = []):
        self.path = os.path.join(EXAMPLES_ROOT, *path_components)
        self.readme = os.path.join(self.path, "README.md")
        self.artifacts = [os.path.join(self.path, name) for name in artifact_names]
        # Ensures no files in addition to the specified artifacts were created.
        self.original_files = []

    def _get_file_list(self):
        return [path for path in glob.iglob(os.path.join(self.path, "*")) if "__pycache__" not in path]

    def _remove_artifacts(self, must_exist=True):
        for artifact in self.artifacts:
            if must_exist:
                print(f"Checking for the existence of artifact: {artifact}")
                assert os.path.exists(artifact), f"{artifact} does not exist!"
            elif not os.path.exists(artifact):
                continue

            if os.path.isdir(artifact):
                shutil.rmtree(artifact)
            else:
                os.remove(artifact)

    def __enter__(self):
        self._remove_artifacts(must_exist=False)

        self.original_files = self._get_file_list()
        return helper.consolidate_code_blocks_from_readme(self.readme)

    def run(self, block, sandboxed_install_run):
        return sandboxed_install_run(block, cwd=self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Checks for and removes artifacts expected by this example
        """
        self._remove_artifacts()
        assert (
            self._get_file_list() == self.original_files
        ), "Unexpected files were created. If this is the desired behavior, add the file paths to `artifact_names`"

    def __str__(self):
        return os.path.relpath(self.path, EXAMPLES_ROOT)


EXAMPLES = [Example(["nanogpt"]), Example(["diffusion"])]


@pytest.mark.l1
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda case: str(case))
def test_examples(example, sandboxed_install_run):
    with open(example.readme, "r", encoding="utf-8") as f:
        contents = f.read()
        # Check that the README has all the expected sections.
        assert "## Introduction" in contents, "All example READMEs should have an 'Introduction' section!"
        assert "## Running The Example" in contents, "All example READMEs should have a 'Running The Example' section!"

    statuses = []
    with example as command_blocks:
        # NOTE: This logic is not smart enough to handle multiple separate commands in a single block.
        for block in command_blocks:
            if block.has_marker("test: ignore") or not block.has_marker("command"):
                continue

            block_text = str(block)
            if block.has_marker("test: expected_stdout"):
                print("Checking command output against expected output:")
                assert re.match(dedent(block_text).strip(), statuses[-1].stdout.strip())
            else:
                status = example.run(block_text, sandboxed_install_run)

                details = f"Note: Command was: {block_text}.\n==== STDOUT ====\n{status.stdout}\n==== STDERR ====\n{status.stderr}"
                if block.has_marker("test: xfail"):
                    assert not status.success, f"Command that was expected to fail did not fail. {details}"
                else:
                    assert status.success, f"Command that was expected to succeed did not succeed. {details}"
                statuses.append(status)
