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

import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Flip


class TestFlip:
    @pytest.mark.parametrize(
        "dims",
        [None, [], 0, 1, -1, [0], [1], [-1], [0, 1]],
    )
    def test_flip_properties(self, dims):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        f = tp.flip(t, dims=dims)
        assert isinstance(f, tp.Tensor)
        assert isinstance(f.trace_tensor.producer, Flip)
        assert f.trace_tensor.rank == 2
        assert f.shape == [2, 5]

    def test_flip_0_rank(self):
        t = tp.Tensor(1)
        f = tp.flip(t)
        assert isinstance(f, tp.Tensor)
        assert isinstance(f.trace_tensor.producer, Flip)
        assert f.trace_tensor.rank == 0

    def test_out_of_range_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException,
            match=r"All dimensions for flip must be in the range \[-2, 2\), but dimension 3 is out of range",
        ):
            tp.flip(t, dims=3)

    def test_repeated_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException, match="All dimensions for flip must be unique but dimension 0 is repeated"
        ):
            tp.flip(t, dims=[0, 1, 0])

    def test_out_of_range_negative_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException,
            match=r"All dimensions for flip must be in the range \[-2, 2\), but dimension -3 is out of range",
        ):
            tp.flip(t, dims=-3)

    def test_repeated_negative_dim(self):
        t = tp.Tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with helper.raises(
            tp.TripyException, match=r"All dimensions for flip must be unique but dimension 1 \(-1\) is repeated"
        ):
            tp.flip(t, dims=[0, 1, -1])

    def test_flip_rank_0_with_dims(self):
        t = tp.Tensor(1)
        with helper.raises(tp.TripyException, match="It is not possible to flip a rank-0 tensor"):
            tp.flip(t, dims=0)
