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


EXAMPLES = [
    Example(["nanogpt"]),
    Example(["segment-anything-model-v2"], artifact_names=["truck.jpg", "saved_engines/", "output/", "checkpoints/"]),
]


# We want to test our examples with both the latest commit and public build.
# This is because we always want the examples to work with pip-installable build, but also
# don't want them to break on TOT.
@pytest.mark.l1
@pytest.mark.l1_release_package
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda case: str(case))
def test_examples(example, sandboxed_install_run):

    def test_with_tolerance(expected, actual, tolerance):
        return (abs(float(actual) - float(expected)) / float(expected)) * 100 <= float(tolerance)

    def process_tolerances(expected_output):
        specs = []
        placeholder_regex = r"{(\d+\.?\d*)~(\d+)%}"
        pattern = expected_output

        # Replace tolerance patterns with more flexible capture group
        matches = list(re.finditer(placeholder_regex, pattern))
        for match in matches:
            specs.append((match.group(1), match.group(2)))
            pattern = pattern.replace(match.group(0), r"(\d+\.?\d*)", 1)

        # Escape parentheses but not our capture group
        pattern = pattern.replace("(", r"\(")
        pattern = pattern.replace(")", r"\)")
        pattern = pattern.replace(r"\(\d+\.?\d*\)", r"(\d+\.?\d*)")

        # Make whitespace flexible
        pattern = pattern.replace(" ", r"\s+")

        return pattern.strip(), specs

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

            code = str(block)
            if block.has_marker("test: expected_stdout"):
                out = statuses[-1].stdout.strip()
                # expected = dedent(code).strip()
                expected_outs = dedent(code).split("====")
                for expected in expected_outs:
                    pattern, specs = process_tolerances(expected)

                    # Apply the DOTALL flag to allow `.` to match newlines
                    compiled_pattern = re.compile(pattern, re.DOTALL)
                    match = compiled_pattern.search(out)

                    # match = re.search(pattern, out)
                    if match and specs:
                        # Check if captured numbers are within tolerance
                        matched = all(
                            test_with_tolerance(expected, actual, tolerance)
                            for (expected, tolerance), actual in zip(specs, match.groups())
                        )
                    else:
                        matched = bool(match)

                    if matched:
                        break

                print("matched!" if matched else "did not match!")
                print(f"==== STDOUT ====\n{out}")
                assert matched
            else:
                status = example.run(code, sandboxed_install_run)

                details = (
                    f"Note: Command was: {code}.\n==== STDOUT ====\n{status.stdout}\n==== STDERR ====\n{status.stderr}"
                )
                if block.has_marker("test: xfail"):
                    assert not status.success, f"Command that was expected to fail did not fail. {details}"
                else:
                    assert status.success, f"Command that was expected to succeed did not succeed. {details}"
                statuses.append(status)
