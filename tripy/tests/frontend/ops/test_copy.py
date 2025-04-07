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
from tests import helper


class TestCopy:
    def test_cannot_copy_within_same_device_kind(self):
        cpu_tensor = tp.Tensor(data=[1, 2, 3], device=tp.device("cpu"))
        with helper.raises(tp.TripyException, match="Copying within the same device kind is not currently supported"):
            tp.copy(cpu_tensor, tp.device("cpu"))
