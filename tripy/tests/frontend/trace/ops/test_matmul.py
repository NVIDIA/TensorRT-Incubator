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

import pytest

import numpy as np

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import MatrixMultiplication


class TestMatMul:
    def test_op_func(self):
        a = tp.Tensor(np.arange(6).reshape((2, 3)).astype(np.float32))
        b = tp.Tensor(np.arange(6).reshape((2, 3))[::-1].astype(np.float32))
        out = a @ b
        assert isinstance(a, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, MatrixMultiplication)

    def test_0d_matrix_fails(self):
        a = tp.ones(tuple(), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)

        with helper.raises(
            tp.TripyException, match="Input tensors must have at least 1 dimension.", has_stack_info_for=[a, b]
        ):
            c = a @ b
            c.eval()

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float16)

        with helper.raises(
            tp.TripyException, match="Mismatched data types for '__matmul__'.", has_stack_info_for=[a, b]
        ):
            c = a @ b

    def test_incompatible_1d_shapes_fails(self):
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = a @ b

        with helper.raises(
            tp.TripyException, match="contracting dimension sizes must match for lhs/rhs", has_stack_info_for=[a, b, c]
        ):
            c.eval()

    def test_incompatible_2d_shapes_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((3, 6), dtype=tp.float32)
        c = a @ b

        with helper.raises(
            tp.TripyException, match="contracting dimension sizes must match for lhs/rhs", has_stack_info_for=[a, b, c]
        ):
            c.eval()

    @pytest.mark.parametrize(
        "a, b, expected_rank",
        [
            (
                tp.ones((2,)),
                tp.ones((2,)),
                0,
            ),
            (tp.ones((2, 3)), tp.ones((3, 2)), 2),
            (tp.ones((4, 2, 3)), tp.ones((3, 2)), 3),
        ],
    )
    def test_infer_rank(self, a, b, expected_rank):
        out = a @ b
        assert out.trace_tensor.rank == expected_rank
