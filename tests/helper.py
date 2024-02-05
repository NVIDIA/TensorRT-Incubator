import contextlib
import importlib
import inspect
import io
import logging
import os
import pkgutil
from textwrap import dedent, indent
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest
import torch

import tripy as tp
from tripy.common.exception import _make_stack_info_message
from tripy.common.logging import G_LOGGER, set_logger_mode
from tripy.frontend import Tensor
from tripy.frontend.trace import Trace

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))


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


def all_same(a: List[int] or List[float], b: List[int] or List[float]):
    """
    Compare two lists element-wise for equality.

    Args:
        a (list): The first list.
        b (list): The second list.

    Returns:
        bool: True if the lists have the same elements in the same order, False otherwise.
    """
    if len(a) != len(b):
        return False

    for a, b in zip(a, b):
        if a != b:
            return False

    return True


class CodeBlock(str):
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
        if not in_code_block:
            out.append(line)

        if in_code_block:
            # Special directives can be used in the code blocks, but should not be
            # made part of our CodeBlock objects. Instead, we append them just before
            # the actual code, which will be right after the `.. code-block:: python` line.
            def is_directive(line):
                if not line.strip().startswith(":"):
                    return False
                tokens = line.strip().split(" ")
                if not tokens:
                    return False
                return tokens[0].endswith(":")

            if is_directive(line):
                out.insert(-1, line)
            # If the line is empty or starts with whitespace, then we're still in the code block.
            elif not line or line.lstrip() != line:
                out[-1] = CodeBlock(out[-1] + line + "\n")
            else:
                out.append(line)
                in_code_block = False

        # This cannot be an `else` statement or we'd discard a line.
        if not in_code_block and line.strip().startswith(".. code-block:: python"):
            in_code_block = True
            out.append(CodeBlock())

    return out


def exec_doc_example(code) -> Dict[str, Any]:
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

        blocks = [dedent(block) for block in consolidate_code_blocks(obj.__doc__) if isinstance(block, CodeBlock)]
        if blocks is None:
            print(f"Skipping {get_qualname(obj)} because no example was present in the docstring")
            continue

        docstrings.extend(blocks)
        ids.extend([f"{get_qualname(obj)}:{idx}" for idx in range(len(blocks))])

    return docstrings, ids
