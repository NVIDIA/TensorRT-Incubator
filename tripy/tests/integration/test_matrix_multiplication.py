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

import cupy as cp
import numpy as np
import pytest

import tripy as tp
import tripy.common.datatype


# Provided since we cannot perform something like `compile_fixture(@, a, b)` or `compile_fixture(__matmul__, a, b)`
def matmul(a, b):
    return a @ b


class TestMatrixMultiplication:

    def test_2d_tensors(self, compile_fixture):
        a_np = np.arange(6).reshape((2, 3)).astype(np.float32)
        b_np = np.arange(6).reshape((3, 2)).astype(np.float32)
        a = tp.Tensor(a_np)
        b = tp.Tensor(b_np)

        out = compile_fixture(matmul, a, b)
        assert tp.allclose(out, tp.Tensor(a_np @ b_np))

    def test_1d_tensors(self, compile_fixture):
        a_np = np.arange(64).astype(np.float32)  # 1D Tensor
        b_np = np.arange(64).astype(np.float32)  # 1D Tensor
        a = tripy.Tensor(cp.asanyarray(a_np))
        b = tripy.Tensor(cp.asanyarray(b_np))

        out = compile_fixture(matmul, a, b)
        assert tp.allclose(out, tp.Tensor(cp.array(a_np @ b_np)), atol=1e-2)

    @pytest.mark.parametrize(
        "shape_a, shape_b",
        [
            ((3,), (3, 2)),  # 1D Tensor and 2D tensor
            ((3, 2), (2,)),  # 2D Tensor and 1D tensor
            ((2, 3, 4), (4, 2)),  # 3D tensor and 2D tensor
            ((3, 2, 3, 4), (4, 2)),  # 4D tensor and 2D tensor
            ((3, 2, 3), (1, 3, 2)),  # Broadcasting batch dimension
            ((1, 2, 3), (0, 0, 3, 2)),  # Broadcasting batch dims with 0
        ],
    )
    def test_broadcast_gemm(self, shape_a, shape_b, compile_fixture):
        a_np = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
        b_np = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)
        a = tp.Tensor(a_np)
        b = tp.Tensor(b_np)

        out = compile_fixture(matmul, a, b)
        assert tp.allclose(out, tp.Tensor(a_np @ b_np))
