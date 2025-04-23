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

import dataclasses
import functools
import hashlib
import inspect
import math
import os
import time
import typing
from typing import Any, List, Optional, Sequence, Tuple, Union

from colored import Fore, Style
from nvtripy import constants
from nvtripy.common.exception import raise_error
from nvtripy.logging import logger
from collections import defaultdict


def default(value, default):
    """
    Returns a specified default value if the provided value is None.

    Args:
        value : The value.
        default : The default value to use if value is None.

    Returns:
        object: Either value, or the default.
    """
    return value if value is not None else default


def pascal_to_snake_case(inp):
    return "".join(f"_{c.lower()}" if c.isupper() else c for c in inp).lstrip("_")


def call_once(func):
    """
    Decorator that makes it so that the decorated function can only be called once.
    Any subsequent calls will do nothing.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if wrapper.never_run:
            wrapper.never_run = False
            return func(*args, **kwargs)

    wrapper.never_run = True
    return wrapper


def log_time(func):
    """
    Provides a wrapper for any arbitrary function to measure and log time to execute this function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.timing(f"{func.__name__} executed in {end - start:.4f} seconds")
        return result

    return wrapper


def code_pretty_str(code, filename=None, line_no=None, func=None, enable_color=True):
    def apply_color(inp, color):
        if not enable_color:
            return inp
        return f"{color}{inp}{Style.reset}"

    line_info = ""
    if filename is not None:
        assert (
            line_no is not None and func is not None
        ), f"If file information is provided, line number and function must also be set."
        line_info = f"--> {apply_color(filename, Fore.yellow)}:{line_no} in {apply_color(func + '()', Fore.cyan)}"

    if not code:
        return line_info

    INDENTATION = 6

    def make_line_no_str(index):
        if line_no is None:
            return " " * INDENTATION
        return f"{index + line_no:>{INDENTATION - 1}} "

    line_numbered_code = "\n".join(
        f"{make_line_no_str(index)}| {code_line}" for index, code_line in enumerate(code.splitlines())
    )
    indent = " " * INDENTATION

    return f"{line_info}\n{line_numbered_code}\n{indent}| "


def make_list(obj):
    """
    Ensure the given object is a list. If it's not, convert it into a list.

    Args:
        obj: The object to be converted into a list if necessary.
    Returns:
        A list.
    """
    if isinstance(obj, tuple):
        return list(obj)

    if not isinstance(obj, list) and obj is not None:
        return [obj]
    return obj


def make_tuple(obj):
    """
    Ensure the given object is a tuple. If it's not, convert it into a tuple.

    Args:
        obj: The object to be converted into a tuple if necessary.
    Returns:
        A tuple.
    """
    if isinstance(obj, list):
        return tuple(obj)

    if not isinstance(obj, tuple) and obj is not None:
        return (obj,)
    return obj


def list_to_tuple(nested_list):
    """
    Recursively converts a nested list structure into a nested tuple structure.
    """
    if isinstance(nested_list, list):
        # Recursively apply the conversion to each element in the list
        return tuple(list_to_tuple(item) for item in nested_list)
    else:
        # Return the item as it is if it's not a list
        return nested_list


def should_omit_constant_in_str(shape):
    return math.prod(shape) >= constants.CONSTANT_IR_PRINT_VOLUME_THRESHOLD


def get_dataclass_fields(obj: Any, BaseClass: type) -> List[dataclasses.Field]:
    """
    Returns all dataclass fields of the specified object, excluding fields inherited from BaseClass.
    """
    base_fields = {base_field.name for base_field in dataclasses.fields(BaseClass)}
    return [field for field in dataclasses.fields(obj) if field.name not in base_fields]


##
## Files
##


def warn_if_wrong_mode(file_like: typing.IO, mode: str):
    def binary(mode):
        return "b" in mode

    def readable(mode):
        return "r" in mode or "+" in mode

    def writable(mode):
        return "w" in mode or "a" in mode or "+" in mode

    fmode = file_like.mode
    if (
        binary(fmode) != binary(mode)
        or (readable(mode) and not readable(fmode))
        or (writable(mode) and not writable(fmode))
    ):
        logger.warning(
            f"File-like object has a different mode than requested!\n"
            f"Note: Requested mode was: {mode} but file-like object has mode: {file_like.mode}"
        )


def is_file_like(obj: Any) -> bool:
    try:
        obj.read
        obj.write
    except AttributeError:
        return False
    else:
        return True


def makedirs(path: str):
    dir_path = os.path.dirname(path)
    if dir_path:
        dir_path = os.path.realpath(dir_path)
        if not os.path.exists(dir_path):
            logger.info(f"{dir_path} does not exist, creating now.")
        os.makedirs(dir_path, exist_ok=True)


def load_file(src: Union[str, typing.IO], mode: str = "rb", description: str = None) -> Union[str, bytes, None]:
    """
    Reads from the specified source path or file-like object.

    Args:
        src: The path or file-like object to read from.
        mode: The mode to use when reading.
        description: A description of what is being read.

    Returns:
        The contents read.

    Raises:
        Exception: If the file or file-like object could not be read.
    """
    if description is not None:
        logger.info(f"Loading {description} from {src}")

    if is_file_like(src):
        warn_if_wrong_mode(src, mode)
        # Reset cursor position after reading from the beginning of the file.
        prevpos = src.tell()
        if src.seekable():
            src.seek(0)
        contents = src.read()
        if src.seekable():
            src.seek(prevpos)
        return contents
    else:
        with open(src, mode) as f:
            return f.read()


def save_file(
    contents: Union[str, bytes], dest: Union[str, typing.IO], mode: str = "wb", description: str = None
) -> Union[str, typing.IO]:
    """
    Writes text or binary data to the specified destination path or file-like object.

    Args:
        contents: A string or bytes-like object that can be written to disk.
        dest: The path or file-like object to write to.
        mode: The mode to use when writing.
        description: A description of what is being written.

    Returns:
        The complete file path or file-like object.

    Raises:
        Exception: If the path or file-like object could not be written to.
    """
    if description is not None:
        logger.info(f"Saving {description} to {dest}")

    if is_file_like(dest):
        warn_if_wrong_mode(dest, mode)
        bytes_written = dest.write(contents)
        dest.flush()
        os.fsync(dest.fileno())
        try:
            content_bytes = len(contents.encode())
        except:
            pass
        else:
            if bytes_written != content_bytes:
                logger.warning(
                    f"Could not write entire file. Note: file contains {content_bytes} bytes, "
                    f"but only {bytes_written} bytes were written."
                )
    else:
        makedirs(dest)
        with open(dest, mode) as f:
            f.write(contents)
    return dest


##
## Hashing
##
def md5(*args) -> int:
    """
    Returns the md5sum of an object. This function relies on `repr`
    to generate a byte buffer for the object.
    """
    return int(hashlib.md5(repr(args).encode()).hexdigest(), base=16)


class UniqueNameGen:
    """
    Generates unique names based on inputs and outputs.
    """

    _used_names = defaultdict(int)

    @staticmethod
    def gen_uid(inputs=None, outputs=None):
        while True:
            elems = []

            if inputs:
                elems.append("ins")
                elems.extend(inputs)

            if outputs:
                elems.append("outs")
                elems.extend(outputs)

            elems = "_".join(elems)
            uid = f"{elems}_{UniqueNameGen._used_names[elems]}"
            UniqueNameGen._used_names[elems] += 1
            return uid


##
## Functions
##
def get_positional_arg_names(func, *args) -> Tuple[List[Tuple[str, Any]], Optional[Tuple[str, int]]]:
    # Returns the names of positional arguments by inspecting the function signature.
    # In the case of variadic positional arguments, we cannot determine names, so we use
    # None instead. To assist in further processing, this function also returns the name
    # and start index of the variadic args in a pair if present (None if not).
    signature = inspect.signature(func)
    arg_names = []
    varargs_name = None
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # Positional arguments cannot follow variadic positional arguments
            # (they would just be absorbed into the variadic argument).
            varargs_name = name
            break

        arg_names.append(name)

    # For all variadic positional arguments, assign the name of the variadic group.
    num_variadic_args = len(args) - len(arg_names)
    variadic_start_idx = len(arg_names)
    arg_names.extend([varargs_name] * num_variadic_args)
    return list(zip(arg_names, args)), (varargs_name, variadic_start_idx) if num_variadic_args > 0 else None


def merge_function_arguments(func, *args, **kwargs) -> Tuple[List[Tuple[str, Any]], Optional[Tuple[str, int]]]:
    # Merge positional and keyword arguments, trying to determine names where possible.
    # Also returns a pair containing the variadic arg name and start index if present (None otherwise).
    all_args, var_arg_info = get_positional_arg_names(func, *args)
    all_args.extend(kwargs.items())
    return all_args, var_arg_info
