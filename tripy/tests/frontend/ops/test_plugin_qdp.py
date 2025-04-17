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
import nvtripy as tp
import pytest
from tests import helper


class TestQuicklyDeployablePlugin:
    @pytest.mark.parametrize(
        "plugin_id, err",
        [
            ("fake_id", "Invalid plugin ID: 'fake_id'."),
            ("test::fake_id", "Plugin 'test::fake_id' not found."),
        ],
    )
    def test_invalid_id(self, plugin_id, err):
        inp = tp.iota((2, 2))
        with helper.raises(tp.TripyException, err):
            out = tp.plugin(plugin_id, [inp])

    def test_invalid_attribute(self):
        inp = tp.iota((2, 2))
        with helper.raises(tp.TripyException, "Unexpected attribute: 'fake'."):
            out = tp.plugin("example::elemwise_add_plugin", [inp], fake=1)

    def test_invalid_attribute_type(self):
        inp = tp.iota((2, 2))
        with helper.raises(tp.TripyException, "Unexpected attribute type: 'float'."):
            out = tp.plugin("example::elemwise_add_plugin", [inp], block_size=1.0)
