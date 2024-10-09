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

import tripy as tp


def test_timing_cache(tmp_path):
    # Create a dummy file using pytest fixture for timing cache.
    dummy_file = tmp_path / "dummy.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    def func(a, b):
        c = a + b
        return c

    tp.compile(func, args=[tp.InputInfo((2,), dtype=tp.float32), tp.InputInfo((2,), dtype=tp.float32)])

    assert dummy_file.parent.exists(), "Cache directory was not created."
    assert dummy_file.exists(), "Cache file was not created."

    dummy_file = tmp_path / "dummy1.txt"
    tp.config.timing_cache_file_path = str(dummy_file)

    # Check if new timing cache file is created when we recompile.
    tp.compile(func, args=[tp.InputInfo((1,), dtype=tp.float32), tp.InputInfo((1,), dtype=tp.float32)])

    assert dummy_file.exists(), "Cache file was not created."
    assert tp.config.timing_cache_file_path == str(
        dummy_file
    ), f"get_timing_cache_file() path does not match the user provided path {str(dummy_file)}"
