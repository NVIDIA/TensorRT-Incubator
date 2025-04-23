#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NOTE: We avoid using `inspect` functions as much as possible because they are much slower than
# working directly with the frame.
import sys
from dataclasses import dataclass
from typing import Optional, Tuple


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
    code: Optional[str]
    """Code corresponding to the file and line number. To save space, this is not available for all frames."""
    _dispatch_target: str
    """If this stack frame is from a dispatch function in the function registry, this field indicates which function it's dispatching to"""
    column_range: Optional[Tuple[int, int]]
    """Column range for the line. This is only available in Python 3.11+"""

    def is_user_frame(self):
        # In some cases, there may not be a module, e.g. if using the interactive shell.
        # In that case, we should treat it as an external frame.
        module = self.module or ""
        return "nvtripy" not in module.split(".")

    def fetch_source_code(self):
        if self.code is not None:
            return

        # Note that in some cases, e.g. when code is being provided via the interactive shell, we may not be able to retrieve it.
        # In that case we just leave it empty.
        try:
            lines = open(self.file, "r").readlines()
        except:
            self.code = "<could not fetch source code>"
        else:
            self.code = lines[self.line - 1].rstrip()


class StackInfo(list):
    def __init__(self, lst, include_code_index: Optional[int] = None):
        super().__init__(lst)
        self.include_code_index = include_code_index
        self._code_fetched = False

    def fetch_source_code(self):
        if self._code_fetched:
            return

        for index, source_info in enumerate(self):
            if source_info.is_user_frame():
                source_info.fetch_source_code()
            elif self.include_code_index is not None and index >= self.include_code_index:
                source_info.fetch_source_code()

        self._code_fetched = True

    def get_first_user_frame_index(self) -> Optional[int]:
        for index, source_info in enumerate(self):
            if source_info.is_user_frame():
                return index

    def __repr__(self):
        return "\n".join(map(str, self))


def get_stack_info(include_code_index: int = None) -> StackInfo:
    import nvtripy.utils.function_registry

    stack_info = StackInfo([], include_code_index)

    # Exclude the current stack frame since we don't care about the get_stack_info() function itself.
    frame = sys._getframe().f_back

    MAX_STACK_DEPTH = 100
    for _ in range(MAX_STACK_DEPTH):
        if not frame:
            break

        source_info = SourceInfo(
            module=frame.f_globals.get("__name__"),
            file=frame.f_code.co_filename,
            line=frame.f_lineno,
            function=frame.f_code.co_name,
            code=None,
            _dispatch_target="",
            column_range=None,
        )
        if source_info.module == nvtripy.utils.function_registry.__name__ and source_info.function == "wrapper":
            source_info._dispatch_target = frame.f_locals["key"]

        try:
            frame.f_code.co_positions
        except AttributeError:
            pass
        else:
            # For 3.11+, positions is an iterator of (line, end_line, column, end_column)
            for pos in frame.f_code.co_positions():
                if pos[0] == frame.f_lineno:
                    _, _, start, end = pos
                    source_info.column_range = (start, end)
                    break

        stack_info.append(source_info)
        frame = frame.f_back

    return stack_info


def get_module_names_to_exclude_from_stack_info():
    """
    Returns a set of module names to exclude from stack information when displaying exceptions
    or trying to retrieve column information from code.
    """
    import nvtripy.utils.function_registry as function_registry
    import nvtripy.utils.wrappers as wrappers

    return {mod.__name__ for mod in [function_registry, wrappers]}
