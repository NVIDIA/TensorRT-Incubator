import inspect
from dataclasses import dataclass
from typing import List, NewType


@dataclass
class SourceInfo:
    """
    Describes a location in Python code. For example, this includes file and line information.
    """

    module: str
    """The name of the module"""
    file: str
    """The file path"""
    line: int
    """The line number"""
    function: str
    """The name of the function"""


StackInfo = NewType("StackInfo", List[SourceInfo])


def get_stack_info() -> StackInfo:
    """
    Returns stack information for the current call stack.

    Returns:
        Stack information for the current call stack.
    """
    stack_info = StackInfo([])
    # Exclude the current stack frame since we don't care about the get_stack_info() function itself.
    for frame in inspect.stack()[1:]:
        stack_info.append(
            SourceInfo(
                module=inspect.getmodule(frame.frame).__name__,
                file=frame.filename,
                line=frame.lineno,
                function=frame.function,
            )
        )
    return stack_info
