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

import contextlib
import copy
import glob
import importlib
import inspect
import io
import os
import pkgutil
import re
from textwrap import dedent, indent
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import black
import cupy as cp
import numpy as np
import pytest
import torch

import nvtripy as tp
from nvtripy import utils
from nvtripy.common.exception import str_from_stack_info
from nvtripy.frontend import Tensor
from nvtripy.frontend.trace import Trace

TAB_SIZE = 4

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def get_files_with_extension(ext):
    return [
        path
        for path in glob.glob(os.path.join(ROOT_DIR, "**", f"*{ext}"), recursive=True)
        if not path.startswith(
            (
                os.path.join(ROOT_DIR, "build"),
                os.path.join(ROOT_DIR, "mlir-tensorrt"),
                os.path.join(ROOT_DIR, "stablehlo"),
            )
        )
    ]


MARKDOWN_FILES = get_files_with_extension(".md")

PYTHON_FILES = get_files_with_extension(".py")


@contextlib.contextmanager
def raises(ExcType: type, match: Optional[str] = None, has_stack_info_for: Sequence[tp.Tensor] = None):
    with pytest.raises(ExcType, match=match) as exc_info:
        yield exc_info

    error_msg = str(exc_info.value)
    print(error_msg)

    error_msg = dedent(error_msg).strip()
    has_stack_info_for = has_stack_info_for or []
    for tensor in has_stack_info_for:
        # Stack info is indented since it's part of the `details` block in `raise_error`
        expected_stack_info = indent(str_from_stack_info(tensor.stack_info).strip(), " " * 4)
        assert expected_stack_info in error_msg, f"Missing stack information for tensor:\n{expected_stack_info}"


@contextlib.contextmanager
def config(name: str, value: Any):
    """
    Temporarily changes a configuration option.
    """
    old_value = getattr(tp.config, name)
    try:
        setattr(tp.config, name, value)
        yield
    finally:
        setattr(tp.config, name, old_value)


# Supported NumPy data types
NUMPY_TO_TRIPY = {
    bool: tp.bool,
    np.int8: tp.int8,
    np.int32: tp.int32,
    np.int64: tp.int64,
    np.float16: tp.float16,
    np.float32: tp.float32,
    # np.int16,  # TODO(#247): Add support for int16
    # np.uint8,  # TODO(#247): Add support for uint8
    # np.uint16, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    # np.uint32, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    # np.uint64, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    # np.float64,  # TODO(#247): Add support for float64
}

TRIPY_TO_NUMPY = {v: k for k, v in NUMPY_TO_TRIPY.items()}


TORCH_DTYPES = {
    tp.float32: torch.float32,
    tp.float16: torch.float16,
    tp.bfloat16: torch.bfloat16,
}


def get_code_bounds(lines):
    # Returns the start and end index of lines of pure code in a block. The block may contain backticks
    # or RST markup indicating a code block.
    code_start = len(lines)
    code_end = 0
    BLOCK_MARKUP = {"```", ".. code-block::", ":"}
    for index, line in enumerate(lines):
        line = line.strip()
        if line and not any(line.startswith(markup) for markup in BLOCK_MARKUP):
            code_start = min(index, code_start)

        if line != "```":
            code_end = max(index, code_end)
    code_end += 1
    return code_start, code_end


class DocstringCodeBlock(str):
    pass


def consolidate_code_blocks(doc):
    """
    Returns a list containing each line of the docstring as a separate string entry with
    code blocks consolidated into CodeBlock instances.

    For example, you may end up with something like:
    [line0, line1, CodeBlock0, line2, line3, CodeBlock1, ...]
    """
    # NOTE: If you edit the parsing logic here, please also update `tests/README.md`.

    doc = dedent(doc)

    out = []
    in_code_block = False
    for line in doc.splitlines():
        if in_code_block:
            # If the line is empty or starts with whitespace, then we're still in the code block.
            if not line or line.lstrip() != line:
                out[-1] = DocstringCodeBlock(out[-1] + line + "\n")
            else:
                in_code_block = False

        # Cannot be an `else` or we'd drop a line.
        if not in_code_block:
            if line.strip().startswith(".. code-block:: python"):
                in_code_block = True
                out.append(DocstringCodeBlock(line + "\n"))
            else:
                out.append(line)

    return out


def exec_code(code, code_locals=None) -> Dict[str, Any]:
    # By default, don't inherit most variables from the current environment
    # so we can be sure the docstring examples work in total isolation.
    code_locals = copy.copy(utils.default(code_locals, {}))
    exec(code, {"tp": tp, "np": np, "torch": torch, "cp": cp}, code_locals)
    return code_locals


@contextlib.contextmanager
def capture_output():
    try:
        outfile = io.StringIO()
        with contextlib.redirect_stdout(outfile), contextlib.redirect_stderr(outfile):
            yield outfile
    finally:
        outfile.flush()
        outfile.seek(0)


def discover_modules():
    mods = [tp]
    while mods:
        mod = mods.pop(0)

        yield mod

        if hasattr(mod, "__path__"):
            mods.extend(
                [
                    importlib.import_module(f"{mod.__name__}.{submod.name}")
                    for submod in pkgutil.iter_modules(mod.__path__)
                ]
            )


def discover_tripy_objects():
    for mod in discover_modules():
        yield from [
            obj
            for obj in mod.__dict__.values()
            if hasattr(obj, "__module__")
            and obj.__module__.startswith("nvtripy")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
        ]


##
## Working with READMEs
##


class Marker:
    """
    Represents special markers in example READMEs used to convey information to
    the infrastructure.

    Special markers follow the format:

    <!-- Tripy: <NAME> Start -->

    and:

    <!-- Tripy: <NAME> End -->

    marking the start and end of the block respectively.
    """

    def __init__(
        self, matches_start_func: Callable[[str], bool] = None, matches_end_func: Callable[[str], bool] = None
    ):
        self.matches_start = matches_start_func

        self.matches_end = matches_end_func

    @staticmethod
    def from_name(name: str) -> "Marker":
        return Marker(
            matches_start_func=lambda line: f"Tripy: {name} Start" in line,
            matches_end_func=lambda line: f"Tripy: {name} End" in line,
        )


AVAILABLE_MARKERS = {
    # For command markers, the start marker may be annotated with a language tag, e.g. ```py, so an exact match is too strict.
    "command": Marker(
        matches_start_func=lambda line: "```" in line,
        matches_end_func=lambda line: "```" in line,
    ),
    # Marks an entire block to be ignored by the tests.
    "test: ignore": Marker.from_name("TEST: IGNORE"),
    # Marks an entire block as being expected to fail.
    "test: xfail": Marker.from_name("TEST: XFAIL"),
    # Marks that a block contains the expected output from the immediate previous block.
    "test: expected_stdout": Marker.from_name("TEST: EXPECTED_STDOUT"),
    # Marks that a block should be run under pytest.
    "test: use_pytest": Marker.from_name("TEST: USE_PYTEST"),
    # Indicates that a block should be omitted from the rendered documentation. Such blocks may still be evaluated.
    "doc: omit": Marker.from_name("DOC: OMIT"),
    # Indicates that a block should not be evaluated for the documentation.
    "doc: no_eval": Marker.from_name("DOC: NO_EVAL"),
}


class MarkerTracker:
    """
    Keeps track of active markers in the current README on a line-by-line basis.
    """

    def __init__(self, readme_path: str):
        self.readme_path: str = readme_path
        self.active_markers: Set[Marker] = set()
        self.entering_markers: Set[Marker] = set()  # The markers that we are currently entering
        self.exiting_markers: Set[Marker] = set()  # The markers that we are currently exiting

    def __enter__(self) -> "MarkerTracker":
        self.file = open(self.readme_path, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.file.close()

    def __iter__(self) -> str:
        for line in self.file.readlines():
            stripped_line = line.strip()
            self.entering_markers.clear()
            self.exiting_markers.clear()

            for marker in AVAILABLE_MARKERS.values():
                if marker not in self.active_markers and marker.matches_start(stripped_line):
                    self.active_markers.add(marker)
                    self.entering_markers.add(marker)
                elif marker in self.active_markers and marker.matches_end(stripped_line):
                    self.active_markers.remove(marker)
                    self.exiting_markers.add(marker)

            yield line.rstrip()

    def entering(self, marker):
        return marker in self.entering_markers

    def exiting(self, marker):
        return marker in self.exiting_markers


class ReadmeCodeBlock:
    def __init__(self, markers: Set[Marker], lang: str, line_number: int):
        self.content: str = None
        self.markers = markers
        self.lang = lang
        self.line_number = line_number

    def add(self, line: str):
        if self.content is None:
            self.content = line
        else:
            self.content += f"\n{line}"

    def has_marker(self, name: str):
        return AVAILABLE_MARKERS[name] in self.markers

    def __str__(self):
        content = self.content or ""
        lines = content.splitlines()
        start, end = get_code_bounds(lines)
        return "\n".join(lines[start:end])

    def __bool__(self):
        return bool(self.content)

    # Returns the original raw contents of the block.
    # This will include the backticks that were stripped out by the consolidation function.
    def raw_str(self) -> str:
        return self.content or ""


# Extract any ``` blocks from the README at the specified path
def consolidate_code_blocks_from_readme(readme_path: str) -> List[ReadmeCodeBlock]:
    cmd_blocks = []
    current_block = ReadmeCodeBlock(markers=set(), lang="text", line_number=0)
    with MarkerTracker(readme_path) as tracker:
        previous_markers = copy.copy(tracker.active_markers)
        for index, line in enumerate(tracker):
            # We use copy here so we don't accidentally alias.
            if tracker.entering(AVAILABLE_MARKERS["command"]):
                # Append previous text block before creating a new block for the command.
                cmd_blocks.append(copy.copy(current_block))
                lang = line.strip().partition("```")[-1]
                current_block = ReadmeCodeBlock(markers=copy.copy(tracker.active_markers), lang=lang, line_number=index)
                current_block.add(line)
            elif tracker.exiting(AVAILABLE_MARKERS["command"]):
                current_block.add(line)
                cmd_blocks.append(copy.copy(current_block))
                # Create new text block for contents between command blocks
                current_block = ReadmeCodeBlock(
                    markers=copy.copy(tracker.active_markers), lang="text", line_number=index
                )
            elif tracker.active_markers != previous_markers:
                cmd_blocks.append(copy.copy(current_block))
                # When markers change, create a new text block
                current_block = ReadmeCodeBlock(
                    markers=copy.copy(tracker.active_markers), lang="text", line_number=index
                )
            else:
                current_block.add(line)

            previous_markers = copy.copy(tracker.active_markers)

    if current_block:
        cmd_blocks.append(current_block)

    return cmd_blocks


##
## Evaluating code to show local and output variables
##

ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def process_code_block_for_outputs_and_locals(
    block: str,
    format_contents: Callable[[str, str, str], str],
    err_msg: str = "",
    local_vars: Dict[str, Any] = None,
    strip_assertions: bool = False,
):
    # Make sure to update `docs/README.md` if updating the behavior of this function.
    local_vars = utils.default(local_vars, {})

    TRIPY_CLASSES = [tripy_obj for tripy_obj in discover_tripy_objects() if inspect.isclass(tripy_obj)]
    # Special tags are documented under docs/README.md.
    NO_EVAL = "# doc: no-eval"
    NO_PRINT_LOCALS = "# doc: no-print-locals"
    PRINT_LOCALS = "# doc: print-locals"
    ALLOW_EXCEPTION = "# doc: allow-exception"
    REMOVE_TAGS = [NO_PRINT_LOCALS, PRINT_LOCALS, NO_EVAL, ALLOW_EXCEPTION]
    if strip_assertions:
        REMOVE_TAGS.append("assert ")
    OMIT_COMMENT = "# doc: omit"

    should_append_locals = True
    should_eval = True
    allow_exception = False

    # By default, we print all local variables. If `print_vars` it not empty,
    # then we'll only print those that appear in it.
    print_vars = set()
    # Set of variables *not* to print
    no_print_vars = set()

    stripped_code_block_lines = []  # All code except what was requested to be omitted.
    output_lines = []
    local_var_lines = []

    for block_line in block.splitlines():
        if block_line.strip().startswith(NO_PRINT_LOCALS):
            _, _, names = block_line.strip().partition(NO_PRINT_LOCALS)
            names = list(filter(lambda x: x, names.strip().split(" ")))
            # If no names are specified, then we disable all local variables.
            if not names:
                should_append_locals = False
            else:
                no_print_vars.update(names)

        if block_line.strip() == NO_EVAL:
            should_eval = False

        if block_line.strip() == ALLOW_EXCEPTION:
            allow_exception = True

        if block_line.strip().startswith(PRINT_LOCALS):
            _, _, names = block_line.strip().partition(PRINT_LOCALS)
            print_vars.update(names.strip().split(" "))

        if any(block_line.strip().startswith(tag) for tag in REMOVE_TAGS) or block_line.endswith(OMIT_COMMENT):
            continue

        stripped_code_block_lines.append(block_line)

    # Format the code portion of the block with black. We use a shorter
    # line length so it doesn't overflow in the rendered docs:
    MAX_LINE_LENGTH = 80
    stripped_code_start, stripped_code_end = get_code_bounds(stripped_code_block_lines)
    stripped_code_lines = stripped_code_block_lines[stripped_code_start:stripped_code_end]

    indentation = len(stripped_code_lines[0]) - len(stripped_code_lines[0].lstrip())
    try:
        stripped_code_lines = indent(
            black.format_file_contents(
                dedent("\n".join(stripped_code_lines)), fast=False, mode=black.Mode(line_length=MAX_LINE_LENGTH)
            ),
            prefix=" " * indentation,
        ).splitlines()
    except black.NothingChanged:
        pass

    # Check that comments don't exceed maximum line length. Note that `black` will not automatically split
    # comments, so this needs to be done manually. Without this, each code block will become a scrollable
    # element, making it very annoying to read. It is also annoying to fix this manually, but it is a one
    # time cost and makes the reading experience so much better.
    too_long_lines = []
    for line in stripped_code_lines:
        # The indentation of the code block doesn't show up in the rendered documentation
        # (indentation *within* the block obviously will.)
        if len(line) - indentation > MAX_LINE_LENGTH:
            too_long_lines.append(f">| {line}")
    too_long_lines = "\n".join(too_long_lines)
    assert (
        not too_long_lines
    ), f"{err_msg}One or more lines exceed maximum line length ({MAX_LINE_LENGTH} characters). Note: lines were:\n{too_long_lines}\n"

    stripped_code_block_lines = (
        stripped_code_block_lines[:stripped_code_start]
        + stripped_code_lines
        + stripped_code_block_lines[stripped_code_end:]
    )

    if not should_eval:
        return stripped_code_block_lines, local_var_lines, output_lines, local_vars

    # When we run the code, we need to get the original code, not the strpiped one.
    block_lines = block.splitlines()
    code_start, code_end = get_code_bounds(block_lines)
    code = dedent("\n".join(block_lines[code_start:code_end]))

    with capture_output() as outfile:
        try:
            code_locals = exec_code(code, local_vars)
        except Exception as e:
            if allow_exception:
                # We print the error message here so it can be captured in `outfile`
                # and displayed in the output in cases where we actually expect exceptions.
                print(e)
                code_locals = local_vars
            else:
                print(f"{err_msg}\n" f"Note: Code block was:\n\n{block}")
                raise

    new_locals = {
        key: value for key, value in code_locals.items() if key not in local_vars or value is not local_vars[key]
    }

    # Add local variables as a separate code block
    locals_str = ""
    if should_append_locals:
        for name, obj in code_locals.items():

            def should_print():
                # print_vars/no_print_vars always take precedence over anything else
                if name in print_vars:
                    return True
                elif print_vars or name in no_print_vars:
                    return False

                # By default, only print new variables (print_vars may override this)
                if name not in new_locals:
                    return False

                # Skip over any non-tripy types.
                if not any(isinstance(obj, tripy_obj) for tripy_obj in TRIPY_CLASSES):
                    return False

                EXCLUDE_OBJECTS = []

                if any(isinstance(obj, exclude_obj) for exclude_obj in EXCLUDE_OBJECTS):
                    return False

                return True

            if not should_print():
                continue

            def pretty_str_from_dict(dct):
                if not dct:
                    return r"{}"
                ret = "{\n"
                for key, value in dct.items():
                    ret += indent(f"{key}: {value},\n", prefix=" " * TAB_SIZE)
                ret += "}"
                return ret

            locals_str += f"\n>>> {name}"
            if isinstance(obj, tp.Module):
                locals_str += f"\n{obj}"
                locals_str += f"\n>>> {name}.state_dict()\n{pretty_str_from_dict(obj.state_dict())}"
            elif isinstance(obj, dict):
                locals_str += f"\n{pretty_str_from_dict(obj)}"
            else:
                locals_str += f"\n{obj}"

    def split_block_lines(title, contents, lang="python"):
        line = block.splitlines()[1]
        indentation = len(line) - len(line.lstrip())

        out = (
            indent(
                format_contents(title, contents, lang),
                prefix=" " * (indentation - 4),
            )
            + "\n\n"
        )
        return out.splitlines()

    if locals_str:
        local_var_lines = split_block_lines("", locals_str)

    # Add output as a separate code block.
    stdout = outfile.read() or ""

    if stdout:
        # Strip out ANSI control sequences from output:
        stdout = ANSI_ESCAPE.sub("", stdout)
        output_lines = split_block_lines("Output:", stdout, lang="")

    return stripped_code_block_lines, local_var_lines, output_lines, code_locals
