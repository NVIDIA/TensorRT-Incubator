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

import nvtripy as tp


class TestIota:
    def test_infer_rank(self):
        a = tp.iota((2, 3, 4))
        assert a.trace_tensor.rank == 3


class TestIotaLike:
    def test_infer_rank(self):
        t = tp.Tensor([1, 2, 3])
        a = tp.iota_like(t)
        assert a.trace_tensor.rank == 1
