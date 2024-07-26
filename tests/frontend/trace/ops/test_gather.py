
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

import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Gather


class TestGather:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        index = tp.Tensor(np.zeros(1, dtype=np.int32))
        out = tp.gather(a, 0, index)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, Gather)

    def test_incorrect_dtype(self):
        a = tp.Tensor([[1, 2], [3, 4]], shape=(2, 2))
        index = tp.Tensor(np.zeros(1, dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Index tensor for gather operation should be of int32 type.",
            has_stack_info_for=[a, index],
        ):
            b = tp.gather(a, 0, index)

    @pytest.mark.parametrize("index_shape", [(1,), (2, 2)])
    def test_infer_rank(self, index_shape):
        a = tp.Tensor([1, 2, 3, 4])
        index = tp.Tensor(np.zeros(index_shape, dtype=np.int32))
        out = tp.gather(a, 0, index)
        assert out.trace_tensor.rank == a.rank + len(index_shape) - 1
