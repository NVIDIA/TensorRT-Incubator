import contextlib
import copy
import glob
import importlib
import inspect
import os
import pkgutil
from textwrap import dedent, indent
from typing import Any, Callable, Dict, List, Sequence, Set

import numpy as np
import pytest
import torch

import tripy as tp
from tripy.common.exception import _make_stack_info_message
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))

MARKDOWN_FILES = [
    path
    for path in glob.glob(os.path.join(ROOT_DIR, "**", "*.md"), recursive=True)
    if not path.startswith(
        (
            os.path.join(ROOT_DIR, "build"),
            os.path.join(ROOT_DIR, "mlir-tensorrt"),
            os.path.join(ROOT_DIR, "stablehlo"),
        )
    )
]


@contextlib.contextmanager
def raises(ExcType: type, match: str, has_stack_info_for: Sequence[tp.Tensor] = None):
    with pytest.raises(ExcType, match=match) as exc_info:
        yield exc_info

    error_msg = str(exc_info.value)
    print(error_msg)

    error_msg = dedent(error_msg).strip()
    has_stack_info_for = has_stack_info_for or []
    for tensor in has_stack_info_for:
        # Stack info is indented since it's part of the `details` block in `raise_error`
        expected_stack_info = indent(_make_stack_info_message(tensor._stack_info).strip(), " " * 4)
        assert expected_stack_info in error_msg, f"Missing stack information for tensor:\n{expected_stack_info}"


def check_mlir(mlir, expected):
    # Checks a given MLIR module against a string of the expected program.
    # MLIR indents with 2 spaces; we'll replace it with 4 spaces so that it's
    # easier to write the expected string.
    mlir_str = str(mlir).replace(" " * 2, " " * 4).strip()
    print(f"MLIR:\n{mlir_str}")
    assert mlir_str == dedent(expected).strip()


# Supported NumPy data types
NUMPY_TYPES = [
    np.int8,
    # np.int16,
    np.int32,
    # (38): Add cast operation to support unsupported backend types.
    # np.int64, # TODO: Fails as convert-hlo-to-tensorrt=allow-i64-to-i32-conversion is enabled in the mlir backend. Explore disabling - limitation can due to TRT limitation.
    np.uint8,
    # np.uint16,
    # np.uint32,
    # np.uint64,
    np.float16,
    np.float32,
    # (38): Add cast operation to support unsupported backend types.
    # np.float64, # TODO: How do we support in tripy? May be insert a cast to float32 in case of lossless conversion and error otherwise?
]


def torch_type_supported(data: np.ndarray):
    unsupported_dtypes = [np.uint16, np.uint32, np.uint64]
    return data.dtype not in unsupported_dtypes


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


def exec_code(code) -> Dict[str, Any]:
    # Don't inherit most variables from the current environment so we can be sure the docstring examples
    # work in total isolation.
    new_locals = {}
    exec(code, {"tp": tp, "np": np, "torch": torch}, new_locals)
    return new_locals


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
    the testing infrastructure.

    Special markers follow the format:

    <!-- Tripy Test: <NAME> Start -->

    and:

    <!-- Tripy Test: <NAME> End -->

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
            matches_start_func=lambda line: line == f"<!-- Tripy Test: {name} Start -->",
            matches_end_func=lambda line: line == f"<!-- Tripy Test: {name} End -->",
        )


AVAILABLE_MARKERS = {
    # For command markers, the start marker may be annotated with a language tag, e.g. ```py, so an exact match is too strict.
    "command": Marker(
        matches_start_func=lambda line: line.startswith("```"),
        matches_end_func=lambda line: line == "```",
    ),
    # Marks an entire block to be ignored by the tests.
    "ignore": Marker.from_name("IGNORE"),
    # Marks an entire block as being expected to fail.
    "xfail": Marker.from_name("XFAIL"),
    # Marks that a block contains the expected output from the immediate previous block.
    "expected_stdout": Marker.from_name("EXPECTED_STDOUT"),
    # Marks that a block should be run under pytest.
    "pytest": Marker.from_name("PYTEST"),
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
                if not self.is_in(marker) and marker.matches_start(stripped_line):
                    self.active_markers.add(marker)
                    self.entering_markers.add(marker)
                elif marker.matches_end(stripped_line):
                    self.active_markers.remove(marker)
                    self.exiting_markers.add(marker)

            yield line.rstrip()

    def is_in(self, marker: Marker) -> bool:
        """
        Whether we are currently on a line between the specified start and end marker.
        This will always return False for a line containing the marker itself.
        """
        return marker in self.active_markers and not (self.entering(marker) or self.exiting(marker))

    def entering(self, marker):
        return marker in self.entering_markers

    def exiting(self, marker):
        return marker in self.exiting_markers


class ReadmeCodeBlock:
    def __init__(self, markers: Set[Marker], lang: str):
        self.content: str = None
        self.markers = markers
        self.lang = lang

    def add(self, line: str):
        if self.content is None:
            self.content = line
        else:
            self.content += f"\n{line}"

    def has_marker(self, name: str):
        return AVAILABLE_MARKERS[name] in self.markers

    def __str__(self):
        return dedent(self.content)


# Extract any ``` blocks from the README
def load_command_blocks_from_readme(readme) -> List[ReadmeCodeBlock]:
    cmd_blocks = []
    with MarkerTracker(readme) as tracker:
        for line in tracker:
            # We use copy here so we don't accidentally alias.
            if tracker.entering(AVAILABLE_MARKERS["command"]):
                lang = line.strip().partition("```")[-1]
                current_block = ReadmeCodeBlock(markers=copy.copy(tracker.active_markers), lang=lang)
            elif tracker.exiting(AVAILABLE_MARKERS["command"]):
                cmd_blocks.append(copy.copy(current_block))
            elif tracker.is_in(AVAILABLE_MARKERS["command"]):
                current_block.add(line)

    return cmd_blocks
