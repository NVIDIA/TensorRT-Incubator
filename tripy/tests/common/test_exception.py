#
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from dataclasses import dataclass
from textwrap import dedent

from tests import helper

import tripy as tp
from tripy.common.exception import TripyException, _get_function_file_and_lines, str_from_stack_info, raise_error
from tripy.utils import StackInfo, get_stack_info
from tripy.utils.stack_info import SourceInfo


@dataclass
class ObjWithStackInfo:
    stack_info: StackInfo


class TestRaiseError:
    def test_obj_with_stack_info(self):
        obj = ObjWithStackInfo(get_stack_info())
        # Fow now we'll just test that the file name is included in the error message.
        # This proves that the stack info is being accessed.
        with helper.raises(TripyException, match=f"{__file__}"):
            raise_error("Test message", details=[obj])

    def test_can_determine_column_range(self):
        # This is derived from a simple expression:
        # a = tp.zeros((2, 3)) - tp.ones((2, 4))
        stack_info = StackInfo(
            [
                SourceInfo(
                    module="tripy.frontend.tensor",
                    file="/tripy/tripy/frontend/tensor.py",
                    line=52,
                    function="_finalize",
                    code="",
                    _dispatch_target="",
                    column_range=None,
                ),
                SourceInfo(
                    module="tripy.frontend.tensor",
                    file="/tripy/tripy/frontend/tensor.py",
                    line=74,
                    function="build",
                    code="",
                    _dispatch_target="",
                    column_range=None,
                ),
                SourceInfo(
                    module="tripy.frontend.trace.ops.binary_elementwise",
                    file="/tripy/tripy/frontend/ops/binary_elementwise.py",
                    line=175,
                    function="sub",
                    code="",
                    _dispatch_target="",
                    column_range=None,
                ),
                SourceInfo(
                    module="tripy.frontend.utils",
                    file="/tripy/tripy/frontend/utils.py",
                    line=23,
                    function="wrapper",
                    code="            return func(*new_args, **new_kwargs)",
                    _dispatch_target="",
                    column_range=None,
                ),
                SourceInfo(
                    module="tripy.utils.function_registry",
                    file="/tripy/tripy/utils/function_registry.py",
                    line=143,
                    function="__call__",
                    code="        return self.func(*args, **kwargs)",
                    _dispatch_target="",
                    column_range=None,
                ),
                SourceInfo(
                    module="tripy.utils.function_registry",
                    file="/tripy/tripy/utils/function_registry.py",
                    line=237,
                    function="wrapper",
                    code="                        return self.find_overload(key, args, kwargs)(*args, **kwargs)",
                    _dispatch_target="__sub__",
                    column_range=None,
                ),
                SourceInfo(
                    module="__main__",
                    file="/tripy/tmp.py",
                    line=3,
                    function="<module>",
                    code="a = tp.zeros((2, 3)) - tp.ones((2, 4))",
                    _dispatch_target="",
                    column_range=None,
                ),
            ]
        )

        error_msg = str_from_stack_info(stack_info, enable_color=False)
        assert (
            dedent(
                """
                --> /tripy/tmp.py:3 in <module>()
                      |
                    3 | a = tp.zeros((2, 3)) - tp.ones((2, 4))
                      |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                """
            ).strip()
            in dedent(error_msg).strip()
        )

    def test_wrappers_is_excluded(self):
        from tripy import wrappers

        tensor = tp.ones((2, 3))

        stack_info = tensor.stack_info

        assert any(frame.module == wrappers.__name__ for frame in stack_info)

        # Make sure that no extraneous wrapper code is included
        expected = dedent(
            r"""
                --> [a-z_/\.]+:[0-9]+ in full\(\)
                    |
                [0-9]+ |     return full_impl\(shape, value, dtype, output_rank\)
                    |

                --> [a-z_/\.]+:[0-9]+ in ones\(\)
                    |
                [0-9]+ |     return full\(shape, 1, dtype\)
                    |            ^^^^^^^^^^^^^^^^^^^^^ --- required from here

                --> [a-z_/\.]+:[0-9]+ in test_wrappers_is_excluded\(\)
                    |
                [0-9]+ |         tensor = tp.ones\(\(2, 3\)\)
                    |                  ^^^^^^^^^^^^^^^ --- required from here

                """
        ).strip()

        actual = str_from_stack_info(stack_info, enable_color=False)
        assert re.search(expected, actual) is not None
