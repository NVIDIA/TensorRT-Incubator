import os
from textwrap import dedent, indent
from typing import List
import tripy as tp
import numpy as np

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.pardir))

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

    doc = dedent(doc)

    out = []
    in_block = False
    for line in doc.splitlines():
        if not in_block:
            out.append(line)

        if in_block:
            # If the line is empty or starts with whitespace, then we're still in the code block.
            if not line or line.lstrip() != line:
                out[-1] = CodeBlock(out[-1] + line + "\n")
            else:
                in_block = False
        elif line.strip().startswith("::"):
            in_block = True
            out.append(CodeBlock())

    return out


def exec_doc_example(code):
    # Don't inherit variables from the current environment so we can be sure the docstring examples
    # work in total isolation.
    return exec(code, {"tp": tp, "np": np}, {})
