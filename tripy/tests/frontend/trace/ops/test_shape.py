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


class TestShape:
    def test_shape(self):
        a = tp.ones((3, 4))
        shape_a = a.shape
        assert isinstance(a, tp.Tensor)
        assert isinstance(shape_a, tp.Shape)

    def test_infer_rank(self):
        a = tp.ones((3, 4))
        shape_a = a.shape
        assert shape_a.trace_tensor.rank == 1
