#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from textwrap import dedent
from typing import Sequence
import contextlib
import pytest
from tests import helper, paths
from tests.conftest import HAS_FP8

EXAMPLES_ROOT = os.path.join(paths.ROOT_DIR, "examples")


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

            artifact_found = glob.glob(artifact)

            if must_exist:
                print(f"Checking for the existence of artifact: {artifact}")
                assert artifact_found, f"{artifact} does not exist!"

            for f in artifact_found:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

    def __enter__(self):
        self._remove_artifacts(must_exist=False)

        self.original_files = self._get_file_list()
        return helper.consolidate_code_blocks_from_readme(self.readme)

    def run(self, block, tripy_virtualenv):
        print(f"Running command:\n{block}")
        return tripy_virtualenv.run(block, cwd=self.path, capture=True)

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
    Example(
        ["segment-anything-model-v2"],
        artifact_names=["truck.jpg", "bedroom", "saved_engines_l/", "output/", "checkpoints/*.pt"],
    ),
    Example(["diffusion"], artifact_names=["test_fp16_engines/", "output/"]),
]


def test_all_examples_tested():
    tested_examples = set(os.path.relpath(example.path, EXAMPLES_ROOT) for example in EXAMPLES)
    all_examples = set(d for d in os.listdir(EXAMPLES_ROOT) if os.path.isdir(os.path.join(EXAMPLES_ROOT, d)))
    assert tested_examples == all_examples, "Not all examples are being tested!"


# We want to test our examples with both the latest commit and public build.
# This is because we always want the examples to work with pip-installable build, but also
# don't want them to break on TOT.
@pytest.mark.l1
@pytest.mark.l1_release_package
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda case: str(case))
def test_examples(example, tripy_virtualenv):
    def process_tolerances(expected_output):
        # Adjusts the expected output into a regex that will be more lenient when matching
        # values with tolerances. The actual tolerance checks are done separately.
        tolerance_specs = []
        tolerance_regex = r"{(\d+\.?\d*)~(\d+)%}"

        # Replace tolerance patterns with more flexible capture group
        matches = list(re.finditer(tolerance_regex, expected_output))

        if not matches:
            # If there are no tolerance values, don't modify the expected output:
            return expected_output, tolerance_specs

        for match in matches:
            tolerance_specs.append((match.group(1), match.group(2)))
            expected_output = expected_output.replace(match.group(0), r"(\d+\.?\d*)", 1)

        # Escape parentheses but not our capture group
        expected_output = expected_output.replace("(", r"\(")
        expected_output = expected_output.replace(")", r"\)")
        expected_output = expected_output.replace(r"\(\d+\.?\d*\)", r"(\d+\.?\d*)")

        # Make whitespace flexible
        expected_output = expected_output.replace(" ", r"\s+")

        return expected_output.strip(), tolerance_specs

    with open(example.readme, "r", encoding="utf-8") as f:
        contents = f.read()
        # Check that the README has all the expected sections.
        assert "## Introduction" in contents, "All example READMEs should have an 'Introduction' section!"
        assert "## Running The Example" in contents, "All example READMEs should have a 'Running The Example' section!"

    outputs = []
    with example as command_blocks:
        # NOTE: This logic is not smart enough to handle multiple separate commands in a single block.
        for block in command_blocks:
            if block.has_marker("test: ignore") or not block.has_marker("command"):
                continue

            if block.has_marker("example: if_fp8") and not HAS_FP8:
                continue

            code = str(block)
            if block.has_marker("test: expected_stdout"):
                print("Checking command output against expected output: ", end="")
                actual = outputs[-1].strip()
                expected = dedent(code).strip()

                print("Actual: ", actual)
                print("Expected: ", expected)

                expected, tolerance_specs = process_tolerances(expected)
                # Apply the DOTALL flag to allow `.` to match newlines
                expected = re.compile(expected, re.DOTALL)
                match = expected.search(actual)

                # We always want to check if the text matched what we expected:
                matched = bool(match)
                # Additionally, check that numbers are within tolerance values if they were specified:
                if tolerance_specs:
                    matched = matched and all(
                        (abs(float(actual) - float(expected)) / float(expected)) * 100 <= float(tolerance)
                        for (expected, tolerance), actual in zip(tolerance_specs, match.groups())
                    )

                print("matched!" if matched else "did not match!")
                print(f"==== STDOUT ====\n{actual}")
                assert matched
            else:
                output = ""
                with contextlib.ExitStack() as stack:
                    if block.has_marker("test: xfail"):
                        stack.enter_context(helper.raises(Exception))

                    output = example.run(code, tripy_virtualenv)

                outputs.append(output)
