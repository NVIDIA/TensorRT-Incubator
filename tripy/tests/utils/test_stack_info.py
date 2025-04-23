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

import sys

import pytest

import nvtripy.utils
from nvtripy.utils.stack_info import SourceInfo


class TestGetStackInfo:
    def test_get_stack_info(self):
        def func():
            # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
            expected_line_num = sys._getframe().f_lineno + 1
            return nvtripy.utils.stack_info.get_stack_info(), expected_line_num

        # Make sure these two lines remain adjacent since we need to know the offset to use for the line number.
        expected_outer_line_num = sys._getframe().f_lineno + 1
        stack_info, expected_inner_line_num = func()
        stack_info.fetch_source_code()

        assert stack_info[0] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_inner_line_num,
            function=func.__name__,
            code="            return nvtripy.utils.stack_info.get_stack_info(), expected_line_num",
            _dispatch_target="",
            column_range=None,
        )
        assert stack_info[1] == SourceInfo(
            __name__,
            file=__file__,
            line=expected_outer_line_num,
            function=self.test_get_stack_info.__name__,
            code="        stack_info, expected_inner_line_num = func()",
            _dispatch_target="",
            column_range=None,
        )
