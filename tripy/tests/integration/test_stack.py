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

import numpy as np
import cupy as cp
import pytest
import nvtripy as tp

from tests.helper import raises


class TestStack:
    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [
            ([(2, 3, 4), (2, 3, 4)], 0),
            ([(2, 3, 4), (2, 3, 4)], 1),
            ([(2, 3, 4), (2, 3, 4)], -1),
            ([(2, 3, 4)], 0),
        ],
    )
    def test_stack(self, tensor_shapes, dim, eager_or_compiled):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        out = eager_or_compiled(tp.stack, tensors, dim=dim)

        # Create numpy arrays for comparison
        np_tensors = [np.ones(shape) for shape in tensor_shapes]
        expected = np.stack(np_tensors, axis=dim)

        assert out.shape == tuple(expected.shape)
        assert np.array_equal(cp.from_dlpack(out).get(), expected)

    def test_stack_different_ranks(self, eager_or_compiled):
        tensors = [tp.ones((2, 3)), tp.ones((2, 3, 4))]
        with raises(
            tp.TripyException,
            match="Expected all input tensors to have the same rank.",
        ):
            eager_or_compiled(tp.stack, tensors)

    def test_stack_different_shapes(self):
        a = tp.ones((2, 3))
        b = tp.ones((4, 3))
        with raises(tp.TripyException, match=r"all concat input tensors must have the same dimensions"):
            tp.stack([a, b]).eval()
