#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import hashlib
import inspect
import os
import time
import typing
from typing import Any, List, Sequence, Union

from colored import Fore, Style

from tripy import constants
from tripy.common.exception import raise_error
from tripy.logging import logger


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
        start_time = time.time()
        result = func(*args, **kwargs)
        logger.timing(f"{func.__name__} executed in {time.time() - start_time:.4f} seconds")
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

    return f"{line_info}\n{indent}|\n{line_numbered_code}\n{indent}| "


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


##
## Dims
##


def volume(shape):
    """
    Computes volume of a tensor shape.

    Args:
        shape: The shape of a tensor

    Returns:
        Volume of tensor (float)
    """

    volume = 1
    for s in shape:
        volume *= s
    return volume


def flatten_list(data):
    """
    Flattens a nested list into a single list.
    """
    if isinstance(data, (int, float)):
        # Need to return a list here as array.array require input to be a list.
        return [data]
    flat_list = []
    for element in data:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list


def get_shape(data):
    """
    Find the shape of a nested list.

    Args:
        nested_list (list): The input nested list.

    Returns:
        list: The shape of the nested list.
    """
    shape = []
    if isinstance(data, (int, float)):
        # Return empty list for a scalar.
        return []
    while isinstance(data, (list, tuple)):
        shape.append(len(data))
        if len(data) == 0:
            break
        data = data[0]
    return shape


def should_omit_constant_in_str(shape):
    return volume(shape) >= constants.CONSTANT_IR_PRINT_VOLUME_THRESHOLD


def get_dataclass_fields(obj: Any, BaseClass: type) -> List[dataclasses.Field]:
    """
    Returns all dataclass fields of the specified object, excluding fields inherited from BaseClass.
    """
    base_fields = {base_field.name for base_field in dataclasses.fields(BaseClass)}
    return [field for field in dataclasses.fields(obj) if field.name not in base_fields]


def constant_fields(field_names: Sequence[str]):
    """
    Marks fields as immutable and disallows them from being changed
    once they have been set the first time.

    Args:
        field_names: The names of fields that should be made immutable.
    """

    def constant_fields_impl(cls: type):
        default_init = cls.__init__

        @functools.wraps(default_init)
        def custom_init(self, *args, **kwargs):
            self.__initialized_fields = set()
            return default_init(self, *args, **kwargs)

        default_setattr = cls.__setattr__

        @functools.wraps(default_setattr)
        def custom_setattr(self, name, value):
            if name == "__initialized_fields":
                return object.__setattr__(self, name, value)

            if name in field_names:
                if name in self.__initialized_fields:
                    raise_error(f"Field: '{name}' of class: '{cls.__qualname__}' is immutable!")
                self.__initialized_fields.add(name)

            return default_setattr(self, name, value)

        cls.__init__ = custom_init
        cls.__setattr__ = custom_setattr
        return cls

    return constant_fields_impl


##
## Files
##


def find_file_in_dir(file_name: str, search_directory: str) -> List:
    """
    Search for file_name recursively in the root_directory.

    Args:
        file_name: The file name or pattern with wildcards.
        search_directory: The root directory from where to search for file_name.
    Returns:
        List of absolute path for matching files.
    """
    search_pattern = os.path.join(search_directory, "**", file_name)
    matching_files = glob.glob(search_pattern, recursive=True)
    return matching_files


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

    _used_names = set()
    _counter = 0

    @staticmethod
    def gen_uid(inputs=None, outputs=None):
        while True:
            UniqueNameGen._counter += 1
            uid = []

            if inputs:
                uid.append("ins")
                uid.extend(inputs)

            if outputs:
                uid.append("outs")
                uid.extend(outputs)

            uid.append(str(UniqueNameGen._counter))
            uid = "_".join(uid)

            if uid not in UniqueNameGen._used_names:
                UniqueNameGen._used_names.add(uid)
                return uid


##
## Functions
##
def get_positional_arg_names(func, *args):
    # Returns the names of positional arguments by inspecting the function signature.
    # In the case of variadic positional arguments, we cannot determine names, so we use
    # None instead.
    signature = inspect.signature(func)
    arg_names = []
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            # Positional arguments cannot follow variadic positional arguments
            # (they would just be absorbed into the variadic argument).
            break

        arg_names.append(name)

    arg_names.extend([None] * (len(args) - len(arg_names)))
    return list(zip(arg_names, args))


def merge_function_arguments(func, *args, **kwargs):
    # Merge positional and keyword arguments, trying to determine names where possible.
    all_args = get_positional_arg_names(func, *args)
    all_args.extend(kwargs.items())
    return all_args
