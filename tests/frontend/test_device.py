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

import pytest

import tripy
from tripy.common.exception import TripyException


class TestDevice:
    def test_basic_construction(self):
        device = tripy.device("cpu")
        assert device.kind == "cpu"
        assert device.index == 0

    def test_index_construction(self):
        device = tripy.device("gpu:1")
        assert device.kind == "gpu"
        assert device.index == 1

    def test_invalid_device_kind_is_rejected(self):
        with pytest.raises(TripyException, match="Unrecognized device kind"):
            tripy.device("not_a_real_device_kind")

    def test_negative_device_index_is_rejected(self):
        with pytest.raises(TripyException, match="Device index must be a non-negative integer"):
            tripy.device("cpu:-1")

    def test_non_integer_device_index_is_rejected(self):
        with pytest.raises(TripyException, match="Could not interpret"):
            tripy.device("cpu:hi")
