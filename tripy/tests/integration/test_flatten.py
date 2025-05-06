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
import numpy as np
import cupy as cp
import pytest
import nvtripy as tp


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
    def test_flatten(self, shape, start_dim, end_dim, expected_shape, eager_or_compiled):
        cp_a = cp.arange(np.prod(shape)).reshape(shape).astype(np.float32)
        a = tp.Tensor(cp_a)
        b = eager_or_compiled(tp.flatten, a, start_dim=start_dim, end_dim=end_dim)
        assert b.shape == expected_shape
        assert np.array_equal(np.from_dlpack(tp.copy(b, device=tp.device("cpu"))), cp_a.reshape(expected_shape).get())

    def test_flatten_invalid_dims(self, eager_or_compiled):
        shape = (2, 3, 4)
        with pytest.raises(tp.TripyException, match="`start_dim` cannot be larger than `end_dim`"):
            a = tp.ones(shape)
            # Invalid because end_dim < start_dim
            eager_or_compiled(tp.flatten, a, start_dim=2, end_dim=1)

    def test_flatten_single_dim(self, eager_or_compiled):
        shape = (2, 3, 4)
        a = tp.ones(shape)
        # Flattening a single dimension should not change the output
        b = eager_or_compiled(tp.flatten, a, start_dim=1, end_dim=1)
        assert b.shape == (2, 3, 4)
        assert np.array_equal(np.from_dlpack(tp.copy(b, device=tp.device("cpu"))), np.ones(shape, dtype=np.float32))

    def test_flatten_with_unknown_dims(self, eager_or_compiled):
        a = tp.ones((2, 3, 4, 5))
        b = eager_or_compiled(tp.flatten, a, start_dim=1, end_dim=-1)
        assert np.array_equal(np.from_dlpack(tp.copy(b, device=tp.device("cpu"))), np.ones((2, 60), dtype=np.float32))
