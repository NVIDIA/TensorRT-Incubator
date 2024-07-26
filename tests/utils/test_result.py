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

from tripy.utils import Result

from tests import helper


class TestResult:
    def test_cannot_retrieve_value_of_error(self):
        result: Result[int] = Result.err(["error!"])
        with helper.raises(AssertionError):
            result.value

    def test_cannot_retrieve_error_details_of_ok(self):
        result: Result[int] = Result.ok(0)
        with helper.raises(AssertionError):
            result.error_details
