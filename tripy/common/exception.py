import inspect
from textwrap import indent
from typing import Any, List


class TripyException(Exception):
    pass


def raise_error(summary: str, details: List[Any]):
    """
    Raises a Tripy exception with a formatted message.

    Args:
        summary: A summary of the error message. This will be displayed before any other details.
        details: Details on the error. This function handles objects in this list as follows:
            - If they include a `stack_info` member, then information on the first user frame is displayed,
                including file/line information as well as the line of code.
            - If they are Tripy datatypes, they are pretty-printed.
            - In all other cases, the object is just converted to a string.

    Raises:
        TripyException
    """
    from tripy.common.datatype import dtype

    detail_msg = ""
    for detail in details:
        if hasattr(detail, "stack_info"):
            frame_strs = []

            for frame in detail.stack_info:
                if not frame.code:
                    continue

                line_info = f"{frame.file}:{frame.line}"
                separator = "-" * max(len(line_info), len(frame.code))
                frame_info = f"\n\n| {line_info}\n| {separator}\n| {frame.code}\n\n"
                frame_strs.append(frame_info)

            detail_msg += "Called from: ".join(frame_strs)
        elif inspect.isclass(detail) and issubclass(detail, dtype):
            detail_msg += f"'{detail.name}'"
        else:
            detail_msg += str(detail)

    msg = f"{summary}\n" + indent(detail_msg, " " * 4)
    raise TripyException(msg)
