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

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Permute, Transpose


class TestPermute:
    def test_op_func(self):
        a = tp.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        a = tp.permute(a, (1, 0))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Permute)

    @pytest.mark.parametrize("perm", [(0,), (0, 1, 2)])
    def test_mistmatched_permutation_fails(self, perm):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.permute(a, perm)

        with helper.raises(
            tp.TripyException,
            match="TransposeOp operand rank 2 does not match permutation size",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((3, 2))
        a = tp.permute(a, (1, 0))
        assert a.trace_tensor.rank == 2


class TestTranspose:
    def test_op_func(self):
        a = tp.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
        a = tp.transpose(a, 0, 1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Transpose)

    def test_incorrect_number_of_arguments(self):
        a = tp.ones((2, 3))

        with helper.raises(tp.TripyException, match="Function expects 3 parameters, but 4 arguments were provided."):
            b = tp.transpose(a, 1, 2, 3)

    def test_infer_rank(self):
        a = tp.ones((3, 2))
        a = tp.transpose(a, 0, 1)
        assert a.trace_tensor.rank == 2
