import dataclasses
import glob
import hashlib
import os
import time
import typing
from dataclasses import dataclass
from typing import Any, List, Union

from tripy import config
from tripy.common.logging import G_LOGGER


@dataclass
class ConditionCheck:
    """
    Bundles a boolean with error message details.

    This can be used in cases where we would like to perform a check in some helper, e.g.
    `is_broadcast_compatible` but still display a nice error message with low level feedback
    from said helper (in this example, that could be details on which dimensions are not compatible).

    The caller can access the `details` field for more information on the error message and
    provide it to `raise_error`.
    """

    value: bool
    details: List[str]

    def __bool__(self) -> bool:
        return self.value


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


def log_time(func):
    """
    Provides a wrapper for any arbitrary function to measure and log time to execute this function.
    """

    def wrapper(*args, **kwargs):
        # Get textual representation of args/kwargs.
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        start_time = time.time()
        result = func(*args, **kwargs)
        G_LOGGER.timing(f"{func.__name__}({signature}) executed in {time.time() - start_time:.4f} seconds")
        return result

    return wrapper


def prefix_with_line_numbers(text: str) -> str:
    """
    Adds prefix line number to text.
    """
    lines = text.split("\n")
    numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


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


def volume(shape):
    """
    Computes volume of a tensor shape.

    Args:
        shape: The shape of a tensor

    Returns:
        Volume of tensor (float)
    """
    from tripy.frontend.ops.utils import to_dims

    volume = 1.0
    for s in to_dims(shape):
        volume *= s.max
    return volume


def should_omit_constant_in_str(shape):
    return volume(shape) >= config.CONSTANT_IR_PRINT_VOLUME_THRESHOLD


def get_dataclass_fields(obj: Any, BaseClass: type) -> List[dataclasses.Field]:
    """
    Returns all dataclass fields of the specified object, excluding fields inherited from BaseClass.
    """
    base_fields = {base_field.name for base_field in dataclasses.fields(BaseClass)}
    return [field for field in dataclasses.fields(obj) if field.name not in base_fields]


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
        G_LOGGER.warning(
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
            G_LOGGER.info(f"{dir_path} does not exist, creating now.")
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
        G_LOGGER.info(f"Loading {description} from {src}")

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
        G_LOGGER.info(f"Saving {description} to {dest}")

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
                G_LOGGER.warning(
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
