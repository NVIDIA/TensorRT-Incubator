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

import nvtripy as tp
import pytest
from tests import helper


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [None, [], 0, 1, -1, [0], [1], [-1], [0, 1]],
    )
    def test_flip_properties(self, dims):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        f = tp.flip(t, dim=dims)
        assert isinstance(f, tp.Tensor)
        assert f.trace_tensor.rank == 2
        assert f.shape == (2, 5)

    def test_flip_0_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert isinstance(f, tp.Tensor)
        assert f.trace_tensor.rank == 0

    @pytest.mark.parametrize("dim", [3, -3])
    def test_out_of_range_dim(self, dim):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(tp.TripyException, match=r"Dimension argument is out of bounds"):
            tp.flip(t, dim=dim)

    @pytest.mark.parametrize("dim", [[0, 1, 0], [0, 1, -1]])
    def test_repeated_dim(self, dim):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(tp.TripyException, match="Each dimension may only be specified once,"):
            tp.flip(t, dim=dim)
