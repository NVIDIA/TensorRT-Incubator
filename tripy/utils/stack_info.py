import inspect
from dataclasses import dataclass


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


def _is_user_frame(stack_info):
    # In some cases, there may not be a module, e.g. if using the interactive shell.
    # In that case, we should treat it as an external frame.
    module = stack_info.module or ""
    return "tripy" not in module.split(".")


class StackInfo(list):
    def get_first_user_frame(self):
        for stack_info in self:
            if _is_user_frame(stack_info):
                return stack_info


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
    stack_info = StackInfo([])
    # Exclude the current stack frame since we don't care about the get_stack_info() function itself.
    stack = inspect.stack()[1:]
    first_user_frame_found = False
    for index, frame in enumerate(stack):
        module = inspect.getmodule(frame.frame)

        stack_info.append(
            SourceInfo(
                module=module.__name__ if module else None,
                file=frame.filename,
                line=frame.lineno,
                function=frame.function,
                code="",
            )
        )

        def add_code():
            # Note that in some cases, e.g. when code is being provided via the interactive shell, we may not be able to retrieve it.
            # In that case we just leave it empty.
            stack_info[-1].code = frame.code_context[0].rstrip() if frame.code_context else ""

        if not first_user_frame_found:
            # Only include code up to the first user frame.
            if _is_user_frame(stack_info[-1]):
                first_user_frame_found = True
                add_code()
            elif include_code_index is not None and index >= include_code_index:
                add_code()

    return stack_info
