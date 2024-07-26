
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

import inspect
from dataclasses import dataclass
from textwrap import indent
from typing import Any, List, Tuple, Optional

from colored import Fore, Style

from tripy import export, utils


@export.public_api(document_under="exception")
class TripyException(Exception):
    """
    Base class for exceptions thrown by Tripy.
    """

    pass


# This custom class lets us override the default hint generated by the Python interpreter
# for attribute errors (e.g. `did you mean ...?`). Since we inherit from `AttributeError`,
# users can still catch this just like a regular AttributeError, hence this class does not
# need to be publicly visible.
class TripyAttributeError(AttributeError):
    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


@dataclass
class OmitStackInfo:
    """
    Allows suppression of stack information in error messages by hiding
    the `.stack_info` field. Acts as a passthrough for str/repr.
    """

    obj: Any

    def __str__(self) -> str:
        return str(self.obj)

    def __repr__(self) -> str:
        return repr(self.obj)


def str_from_source_info(source_info, enable_color=True, is_first_frame=True, callee_info=None):
    def apply_color(inp, color):
        if not enable_color:
            return inp
        return f"{color}{inp}{Style.reset}"

    frame_info = ""
    if is_first_frame:
        frame_info += "\n\n"
    pretty_code = utils.code_pretty_str(
        source_info.code, source_info.file, source_info.line, source_info.function, enable_color=enable_color
    )

    frame_info += pretty_code

    column_range = source_info.column_range
    if column_range is None and callee_info is not None:
        # With multiple calls to the same function name on the same line,
        # it is not possible for us to determine which column offset is correct, so we
        # won't include it in that case.
        try:
            candidate_column_offsets = utils.get_candidate_column_offsets(source_info, callee_info)
        except:
            pass
        else:
            if len(candidate_column_offsets) == 1:
                column_range = candidate_column_offsets[0]

    if column_range:
        start, end = column_range
        size = end - start
        frame_info += " " * start + apply_color("^" * (size), Fore.red)
        if not is_first_frame:
            frame_info += " --- required from here"
    else:
        if not is_first_frame:
            frame_info = "Required from:\n" + frame_info
    frame_info += "\n\n"
    return frame_info


def _make_stack_info_message(stack_info: "utils.StackInfo", enable_color: bool = True) -> Optional[str]:
    import tripy.function_registry
    from tripy.frontend.utils import convert_inputs_to_tensors

    EXCLUDE_FUNCTIONS = [convert_inputs_to_tensors]

    def should_exclude(frame):
        for func in EXCLUDE_FUNCTIONS:
            filename = inspect.getsourcefile(func)
            lines, start_line = inspect.getsourcelines(func)

            if frame.file != filename:
                return False

            if frame.line < start_line or frame.line > (start_line + len(lines)):
                return False
            return True

    frame_strs = []
    num_frames_printed = 0
    for index, source_info in enumerate(stack_info):
        if source_info.code is None:
            continue

        if source_info.module == tripy.function_registry.__name__:
            continue

        if should_exclude(source_info):
            continue

        frame_info = str_from_source_info(
            source_info,
            enable_color,
            num_frames_printed == 0,
            callee_info=stack_info[index - 1] if index >= 1 else None,
        )

        frame_strs.append(frame_info)
        num_frames_printed += 1

    if frame_strs:
        return "".join(frame_strs)

    return None


def raise_error(summary: str, details: List[Any] = []):
    """
    Raises a Tripy exception with a formatted message.

    Args:
        summary: A summary of the error message. This will be displayed before any other details.
        details: Details on the error. This function handles objects in this list as follows:
            - If they include a `stack_info` member, then information on the first user frame is displayed,
                including file/line information as well as the line of code.

                IMPORTANT: Any stack frames from the function registry are not displayed since
                the function registry is an implementation detail used to dispatch to the real functions
                we care about. Additionally, any code defined in the functions listed in ``EXCLUDE_FUNCTIONS``
                is omitted.

            - In all other cases, the object is just converted to a string.

    Raises:
        TripyException
    """

    pre_summary = ""
    stack_info = utils.get_stack_info()
    user_frame_index = stack_info.get_first_user_frame_index()
    if user_frame_index is not None:
        pre_summary = str_from_source_info(stack_info[user_frame_index])

    detail_msg = ""
    for detail in details:
        stack_info_message = None
        if hasattr(detail, "stack_info"):
            stack_info_message = _make_stack_info_message(detail.stack_info)
        elif isinstance(detail, utils.StackInfo):
            stack_info_message = _make_stack_info_message(detail)

        if stack_info_message is not None:
            detail_msg += stack_info_message
        else:
            detail_msg += str(detail)

    msg = f"{pre_summary}{summary}\n" + indent(detail_msg, " " * 4)
    # We use `from None` to suppress output from previous exceptions, since we want to handle them internally.
    raise TripyException(msg) from None


def search_for_missing_attr(module_name: str, name: str, look_in: List[Tuple[Any, str]]):
    """
    Searches for an attribute in the given modules/objects and then raises an AttributeError
    including a hint on where to find the attribute.

    This is intended to be called from a module/submodule-level `__getattr__` override.

    Args:
        module_name: The name of the current module or submodule.
        name: The name of the attribute we are searching for.
        look_in: A list containing elements of (obj/submodule, obj_name) to search under.

    Raises:
        AttributeError: This will potentially include a hint if the attribute was found under
            one of the objects in look_in.
    """
    import inspect

    # We look at the call stack to prevent infinite recursion.
    # Consider the case where `tripy` searches under `tripy.XYZ` and vice-versa.
    # If the attribute is not present in either, they will keep ping-ponging back
    # and forth since `hasattr` in this function will call `__getattr__` which will
    # then call `search_for_missing_attr` ad infinitum.
    stack = inspect.stack()

    stack_modules = []
    stack_classes = []
    for frame in stack:
        module = inspect.getmodule(frame.frame)
        if module:
            stack_modules.append(module)

        self_arg = frame.frame.f_locals.get("self")
        if self_arg is not None:
            try:
                class_type = self_arg.__class__
            except:
                pass
            else:
                stack_classes.append(class_type)

    stack_modules = list(filter(lambda mod: mod is not None, [inspect.getmodule(frame.frame) for frame in stack]))
    stack_classes = list([])

    msg = f"Module: '{module_name}' does not have attribute: '{name}'"
    # If a symbol isn't found in the top-level, we'll look at specific classes/modules
    # in case there's a similar symbol there.
    # We provide the names as well since the object name will be the fully qualified name,
    # which is not necessarily what the user uses.

    for obj, obj_name in look_in:
        # Avoid infinite recursion - see comment above.
        if obj in stack_modules + stack_classes:
            if name == "float64":
                msg += f". Did you mean: 'float32'?"
            if name == "int16":
                msg += f". Did you mean: 'int32'?"
            continue

        if hasattr(obj, name):
            msg += f". Did you mean: '{obj_name}.{name}'?"
            break

    raise TripyAttributeError(msg.strip())
