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

import cupy as cp
import numpy as np

import nvtripy as tp


class TestStronglyTyped:
    """
    Sanity test that strongly typed mode is enabled.
    """

    def test_fp16_no_overflow(self):
        a = tp.Tensor([10000.0, 60000.0])
        a = tp.sum(a)  # 7e+4 is out of fp16 upperbound
        a = a / 5.0
        a = tp.cast(a, tp.float16)

        assert cp.from_dlpack(a).get() == np.array([14000], dtype=np.float16)
