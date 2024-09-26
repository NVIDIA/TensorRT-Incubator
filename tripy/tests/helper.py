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

import cupy as cp
import numpy as np
import pytest
import torch

import tripy as tp
from tripy import utils
from tripy.backend.mlir.utils import remove_sym_attr
from tripy.common.exception import _make_stack_info_message
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace

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
        expected_stack_info = indent(_make_stack_info_message(tensor.stack_info).strip(), " " * 4)
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


def check_mlir(mlir, expected):
    # Checks a given MLIR module against a string of the expected program.
    # MLIR indents with 2 spaces; we'll replace it with 4 spaces so that it's
    # easier to write the expected string.
    mlir_str = mlir.operation.get_asm(large_elements_limit=32).replace(" " * 2, " " * 4).strip()
    mlir_str = remove_sym_attr(mlir_str)
    assert mlir_str == dedent(expected).strip()


# Supported NumPy data types
NUMPY_TYPES = [
    np.int8,
    # np.int16,  # TODO(#247): Add support for int16
    np.int32,
    np.int64,
    # np.uint8,  # TODO(#247): Add support for uint8
    # np.uint16, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    # np.uint32, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    # np.uint64, # TODO(#190): Add support for unsupported MLIR-TensorRT types.
    np.float16,
    np.float32,
    # np.float64,  # TODO(#247): Add support for float64
]


def np_to_tripy_dtype(dtype):
    return {
        bool: tp.bool,
        np.int8: tp.int8,
        np.int32: tp.int32,
        np.int64: tp.int64,
        np.float16: tp.float16,
        np.float32: tp.float32,
    }[dtype]


def torch_type_supported(data: np.ndarray):
    unsupported_dtypes = [np.int16, np.uint16, np.uint32, np.uint64]
    return data.dtype not in unsupported_dtypes


TORCH_DTYPES = {
    tp.float32: torch.float32,
    tp.float16: torch.float16,
    tp.bfloat16: torch.bfloat16,
}


class DocstringCodeBlock(str):
    def code(self) -> str:
        # Special directives can be used in the code blocks and they should be
        # excluded from the actual code.
        def is_directive(line):
            if not line.strip().startswith(":"):
                return False
            tokens = line.strip().split(" ")
            if not tokens:
                return False
            return tokens[0].endswith(":")

        text = self.replace(".. code-block:: python", "", 1)
        return "\n".join([line for line in text.splitlines() if not is_directive(line)])


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
            and obj.__module__.startswith("tripy")
            and (inspect.isclass(obj) or inspect.isfunction(obj))
        ]


# In order to test docstrings, we need to recursively discover all submodules
# and any classes/functions contained in those submodules.


# Returns a list of all classes, functions, and methods defined in Tripy.
def get_all_tripy_interfaces():
    all_objects = set()
    for obj in discover_tripy_objects():
        all_objects.add(obj)
        all_objects.update(
            {
                member
                for _, member in inspect.getmembers(
                    obj,
                    lambda member: inspect.isfunction(member)
                    or isinstance(member, property)
                    or inspect.isclass(member),
                )
            }
        )

    # Some sanity checks to make sure we're actually getting all the objects we expect
    assert Tensor in all_objects
    assert Tensor.shape in all_objects
    assert Trace in all_objects

    return all_objects


def get_all_docstrings_with_examples():
    def get_qualname(obj):
        if isinstance(obj, property):
            return obj.fget.__qualname__
        return obj.__qualname__

    # Because of our complicated method registration logic, the free function and method
    # might both be recognized as separate objects by `get_all_tripy_interfaces()`.
    # In order to avoid redundant testing, we compare the docstrings directly instead.
    seen_docstring_hashes = set()
    docstrings = []
    ids = []
    tripy_interfaces = get_all_tripy_interfaces()
    for obj in tripy_interfaces:
        if not obj.__doc__:
            print(f"Skipping {get_qualname(obj)} because no docstring was present")
            continue

        doc_hash = hash(obj.__doc__)
        if doc_hash in seen_docstring_hashes:
            print(f"Skipping {get_qualname(obj)} because it duplicates the docstring of another interface")
            continue
        seen_docstring_hashes.add(doc_hash)

        blocks = [
            dedent(block.code())
            for block in consolidate_code_blocks(obj.__doc__)
            if isinstance(block, DocstringCodeBlock)
        ]
        if blocks is None:
            print(f"Skipping {get_qualname(obj)} because no example was present in the docstring")
            continue

        docstrings.extend(blocks)
        ids.extend([f"{get_qualname(obj)}:{idx}" for idx in range(len(blocks))])

    return docstrings, ids


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
            matches_start_func=lambda line: line == f"<!-- Tripy: {name} Start -->",
            matches_end_func=lambda line: line == f"<!-- Tripy: {name} End -->",
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
    # Indicates that a block should be omitted from the rendered documentation.
    "doc: omit": Marker.from_name("DOC: OMIT"),
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
    def __init__(self, markers: Set[Marker], lang: str):
        self.content: str = None
        self.markers = markers
        self.lang = lang
        self.start_line = ""
        self.end_line = ""

    def add(self, line: str):
        if self.content is None:
            self.content = line
        else:
            self.content += f"\n{line}"

    def has_marker(self, name: str):
        return AVAILABLE_MARKERS[name] in self.markers

    def __str__(self):
        return self.content or ""

    def __bool__(self):
        return bool(self.content)

    # Returns the original raw contents of the block.
    # This will include the backticks that were stripped out by the consolidation function.
    def raw_str(self) -> str:
        contents = str(self)
        if self.lang == "text":
            return contents
        return f"{self.start_line}\n{contents}\n{self.end_line}"


# Extract any ``` blocks from the README at the specified path
def consolidate_code_blocks_from_readme(readme_path: str) -> List[ReadmeCodeBlock]:
    cmd_blocks = []
    current_block = ReadmeCodeBlock(markers=set(), lang="text")
    with MarkerTracker(readme_path) as tracker:
        previous_markers = copy.copy(tracker.active_markers)
        for line in tracker:
            # We use copy here so we don't accidentally alias.
            if tracker.entering(AVAILABLE_MARKERS["command"]):
                # Append previous text block before creating a new block for the command.
                cmd_blocks.append(copy.copy(current_block))
                lang = line.strip().partition("```")[-1]
                current_block = ReadmeCodeBlock(markers=copy.copy(tracker.active_markers), lang=lang)
                current_block.start_line = line
            elif tracker.exiting(AVAILABLE_MARKERS["command"]):
                current_block.end_line = line
                cmd_blocks.append(copy.copy(current_block))
                # Create new text block for contents between command blocks
                current_block = ReadmeCodeBlock(markers=copy.copy(tracker.active_markers), lang="text")
            elif tracker.active_markers != previous_markers:
                cmd_blocks.append(copy.copy(current_block))
                # When markers change, create a new text block
                current_block = ReadmeCodeBlock(markers=copy.copy(tracker.active_markers), lang="text")
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
    code: str,
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

    code_block_lines = []
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

        code_block_lines.append(block_line)

    if not should_eval:
        return code_block_lines, local_var_lines, output_lines, local_vars

    code = dedent(code)

    with capture_output() as outfile:
        try:
            code_locals = exec_code(code, local_vars)
        except Exception as e:
            if allow_exception:
                print(f"Exception occurred: {str(e)}")
                code_locals = local_vars
            else:
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
                locals_str += f".state_dict()\n{pretty_str_from_dict(obj.state_dict())}"
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

    return code_block_lines, local_var_lines, output_lines, code_locals
