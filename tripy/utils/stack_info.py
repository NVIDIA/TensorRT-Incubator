import inspect
from dataclasses import dataclass
from typing import Tuple, Optional


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
    code: str
    """Code corresponding to the file and line number. To save space, this is not available for all frames."""
    _dispatch_target: str
    """If this stack frame is from a dispatch function in the function registry, this field indicates which function it's dispatching to"""
    column_range: Optional[Tuple[int, int]]
    """Column range for the line. This is only available in Python 3.11+"""

    def is_user_frame(self):
        # In some cases, there may not be a module, e.g. if using the interactive shell.
        # In that case, we should treat it as an external frame.
        module = self.module or ""
        return "tripy" not in module.split(".")


class StackInfo(list):
    def get_first_user_frame_index(self) -> int:
        for index, source_info in enumerate(self):
            if source_info.is_user_frame():
                return index

    def __repr__(self):
        return "\n".join(map(str, self))


def get_stack_info(include_code_index: int = None) -> StackInfo:
    """
    Returns stack information for the current call stack.

    Args:
        include_code_index: The index of a frame after which to include code.
                Code is only included up to the first user frame.
                If this index is past the first user frame, then code is only included for the user frame.

    Returns:
        Stack information for the current call stack.
    """
    import tripy.function_registry

    stack_info = StackInfo([])
    # Exclude the current stack frame since we don't care about the get_stack_info() function itself.
    stack = inspect.stack()[1:]

    first_user_frame_found = False

    for index, frame in enumerate(stack):
        module = inspect.getmodule(frame.frame)

        source_info = SourceInfo(
            module=module.__name__ if module else None,
            file=frame.filename,
            line=frame.lineno,
            function=frame.function,
            code="",
            _dispatch_target="",
            column_range=None,
        )
        if source_info.module == tripy.function_registry.__name__ and source_info.function == "wrapper":
            source_info._dispatch_target = frame.frame.f_locals.get("key", "")

        def add_code():
            # Note that in some cases, e.g. when code is being provided via the interactive shell, we may not be able to retrieve it.
            # In that case we just leave it empty.
            source_info.code = frame.code_context[0].rstrip() if frame.code_context else ""

            try:
                # In Python 3.11, frames contain column offset information.
                frame.positions
            except AttributeError:
                pass
            else:
                source_info.column_range = (frame.positions.col_offset, frame.positions.end_col_offset)

        if not first_user_frame_found:
            if source_info.is_user_frame():
                add_code()
                first_user_frame_found = True
            elif include_code_index is not None and index >= include_code_index:
                add_code()

        stack_info.append(source_info)

    return stack_info
