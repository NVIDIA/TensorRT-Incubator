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

import math

import numpy as np
import nvtripy as tp
import pytest


def gemm(a, b):
    return a @ b


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((3,), (3,)),  # 1D tensor and 1D tensor
        ((3,), (3, 2)),  # 1D tensor and 2D tensor
        ((3, 2), (2,)),  # 2D tensor and 1D tensor
        ((2, 3), (3, 2)),  # 2D tensor and 2D tensor
        ((2, 3, 4), (4, 2)),  # 3D tensor and 2D tensor
        ((3, 2, 3, 4), (4, 2)),  # 4D tensor and 2D tensor
        ((3, 2, 3), (1, 3, 2)),  # Broadcasting batch dimension
        ((1, 2, 3), (0, 0, 3, 2)),  # Broadcasting with empty tensor
    ],
)
def test_matmul(shape_a, shape_b, eager_or_compiled):
    a_np = np.arange(np.prod(shape_a)).reshape(shape_a).astype(np.float32)
    b_np = np.arange(np.prod(shape_b)).reshape(shape_b).astype(np.float32)
    a = tp.Tensor(a_np)
    b = tp.Tensor(b_np)

    out = eager_or_compiled(gemm, a, b)
    ref_out = a_np @ b_np
    if not isinstance(ref_out, np.ndarray):
        ref_out = np.array(ref_out)

    assert out.shape == tuple(ref_out.shape)
    if math.prod(out.shape) != 0:
        # TensorRT doesn't currently allow reductions on empty tensors
        assert tp.allclose(out, tp.Tensor(ref_out))
