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
            user_frame = detail.stack_info.get_first_user_frame()
            line_info = f"{user_frame.file}:{user_frame.line}"
            separator = "-" * max(len(line_info), len(user_frame.code))
            detail_msg += f"\n\n| {line_info}\n| {separator}\n| {user_frame.code}\n\n"
        elif inspect.isclass(detail) and issubclass(detail, dtype):
            detail_msg += f"'{detail.name}'"
        else:
            detail_msg += str(detail)

    msg = f"{summary}\n" + indent(detail_msg, " " * 4)
    raise TripyException(msg)
