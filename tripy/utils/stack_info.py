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
    """Code corresponding to the file and line number. To save space, this is only available for the first user frame in the stack."""


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


def get_stack_info() -> StackInfo:
    """
    Returns stack information for the current call stack.

    Returns:
        Stack information for the current call stack.
    """
    stack_info = StackInfo([])
    # Exclude the current stack frame since we don't care about the get_stack_info() function itself.
    stack = inspect.stack()[1:]
    first_user_frame_found = False
    for frame in stack:
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

        if not first_user_frame_found and _is_user_frame(stack_info[-1]):
            first_user_frame_found = True
            # Only include code for the first user frame
            # Note that in some cases, e.g. when code is being provided via the interactive shell, we may not be able to retrieve it.
            # In that case we just leave it empty.
            stack_info[-1].code = frame.code_context[0].rstrip() if frame.code_context else ""

    return stack_info
