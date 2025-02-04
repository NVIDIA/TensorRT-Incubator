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
from tests import helper


def format_contents(kind, contents, lang):
    return f"{'Output' if kind == helper.BlockKind.OUTPUT else ''}\n{contents}"


class TestProcessCodeBlock:
    def test_non_tripy_types_not_printed_as_locals(self):
        # Non-tripy types should never be shown even if there is no `# doc: no-print-locals`
        block = """
        a = 5
        b = "42"
        """

        _, local_var_lines, _, _ = helper.process_code_block_for_outputs_and_locals(block, format_contents)

        assert not local_var_lines

    def test_no_print_locals(self):
        block = """
        # doc: no-print-locals
        gpu = tp.device("gpu")
        cpu = tp.device("cpu")
        """

        _, local_var_lines, _, _ = helper.process_code_block_for_outputs_and_locals(block, format_contents)

        assert not local_var_lines
