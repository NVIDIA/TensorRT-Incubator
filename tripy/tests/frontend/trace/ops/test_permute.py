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

import numpy as np
import pytest
from tests import helper

import tripy as tp
from tripy.frontend.trace.ops import Permute


class TestPermute:
    def test_op_func(self):
        a = tp.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        a = tp.permute(a, (1, 0))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Permute)

    @pytest.mark.parametrize("perm", [(0,), (0, 1, 2)])
    def test_mistmatched_permutation_fails(self, perm):
        a = tp.ones((2, 3), dtype=tp.float32)

        with helper.raises(
            tp.TripyException,
            match="Invalid permutation.",
        ):
            b = tp.permute(a, perm)

    def test_infer_rank(self):
        a = tp.ones((3, 2))
        a = tp.permute(a, (1, 0))
        assert a.trace_tensor.rank == 2
