import time
import glob
import os
from typing import List

from tripy.logging import G_LOGGER


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


def prefix_with_line_numbers(text: str) -> str:
    """
    Adds prefix line number to text.
    """
    lines = text.split("\n")
    numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def ensure_list(obj):
    """
    Ensure the given object is a list. If it's not, convert it into a list.

    Args:
        obj: The object to be converted into a list if necessary.
    Returns:
        A list.
    """
    if not isinstance(obj, list) and obj is not None:
        return [obj]
    return obj
