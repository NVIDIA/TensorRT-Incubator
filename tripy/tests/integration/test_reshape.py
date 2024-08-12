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
import cupy as cp
import pytest
import tripy as tp

from tests import helper


class TestReshape:
    @pytest.mark.parametrize(
        "shape, new_shape",
        [
            ((2, 4), (1, 8)),
            ((2, 4, 8, 9), (8, 8, 9)),
            ((2, 4), (8,)),  # change rank of output
            ((2, 4), (1, -1)),  # check negative dim
        ],
    )
    def test_static_reshape(self, shape, new_shape, compile_fixture):
        cp_a = cp.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        a = tp.Tensor(cp_a, device=tp.device("gpu"))
        b = compile_fixture(tp.reshape, a, new_shape)
        if -1 in new_shape:
            new_shape = tuple(np.prod(shape) // -np.prod(new_shape) if d == -1 else d for d in new_shape)
        assert np.array_equal(cp.from_dlpack(b).get(), cp_a.reshape(new_shape).get())

    def test_invalid_neg_dim_reshape(self, compile_fixture):
        shape = (1, 30)
        new_shape = (-1, -1)
        with helper.raises(tp.TripyException, match="Reshape operation size operand can have only one dimension as -1"):
            a = compile_fixture(tp.reshape, tp.ones(shape), new_shape)
            print(a)

    def test_reshape_shape_tensor(self, compile_fixture):
        a = tp.ones((2, 3, 4))
        b = tp.ones((2, 3, 2, 2))
        out = compile_fixture(tp.reshape, a, (a.shape[0], a.shape[1], b.shape[2], b.shape[3]))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 3, 2, 2), dtype=np.float32))

    def test_reshape_shape_with_unknown(self):
        a = tp.ones((2, 3, 4))
        out = tp.reshape(a, (2, a.shape[1], a.shape[2] / 2, -1))
        assert np.array_equal(cp.from_dlpack(out).get(), np.ones((2, 3, 2, 2), dtype=np.float32))


class TestFlatten:
    @pytest.mark.parametrize(
        "shape, start_dim, end_dim, expected_shape",
        [
            ((2, 3, 4), 0, -1, (24,)),  # Flatten all dimensions
            ((2, 3, 4), 1, -1, (2, 12)),  # Flatten dimensions 1 through end
            ((2, 3, 4), 1, 2, (2, 12)),  # Flatten dimensions 1 through 2
            ((2, 3, 4), 0, 1, (6, 4)),  # Flatten dimensions 0 through 1
            ((2, 3, 4, 5), 1, 3, (2, 60)),  # Flatten dimensions 1 through 3
        ],
    )
    def test_flatten(self, shape, start_dim, end_dim, expected_shape):
        cp_a = cp.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        a = tp.Tensor(cp_a)
        b = tp.flatten(a, start_dim=start_dim, end_dim=end_dim)
        assert b.shape == expected_shape
        assert np.array_equal(cp.from_dlpack(b).get(), cp_a.reshape(expected_shape).get())

    def test_flatten_invalid_dims(self):
        shape = (2, 3, 4)
        with pytest.raises(tp.TripyException, match="Invalid dimensions"):
            a = tp.ones(shape)
            # Invalid because end_dim < start_dim
            tp.flatten(a, start_dim=2, end_dim=1)

    def test_flatten_single_dim(self):
        shape = (2, 3, 4)
        a = tp.ones(shape)
        # Flattening a single dimension should not change the output
        b = tp.flatten(a, start_dim=1, end_dim=1)
        assert b.shape == (2, 3, 4)
        assert np.array_equal(cp.from_dlpack(b).get(), np.ones(shape, dtype=np.float32))

    def test_flatten_with_unknown_dims(self):
        a = tp.ones((2, 3, 4, 5))
        b = tp.flatten(a, start_dim=1, end_dim=-1)
        assert np.array_equal(cp.from_dlpack(b).get(), np.ones((2, 60), dtype=np.float32))
