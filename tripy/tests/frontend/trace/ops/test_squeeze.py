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
import numpy as np

import nvtripy as tp
from tests import helper
from nvtripy.frontend.trace.ops import Squeeze


class TestSqueeze:
    def test_op_func(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        a = tp.squeeze(a, dims=(0, 1))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Squeeze)

    def test_incorrect_dims(self):
        a = tp.Tensor(np.ones((1, 1, 4), dtype=np.int32))
        b = tp.squeeze(a, 2)

        with helper.raises(
            tp.TripyException,
            match=r"number of output elements \(1\) doesn't match expected number of elements \(4\)",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((3, 2, 1, 2))
        b = tp.squeeze(a, 2)
        assert b.trace_tensor.rank == 3
