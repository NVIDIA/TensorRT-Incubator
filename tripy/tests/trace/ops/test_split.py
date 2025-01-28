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

import nvtripy as tp
from tests import helper
from nvtripy.trace.ops import Split


def validate_splits(splits, expected_shapes):
    assert isinstance(splits, list)
    assert len(splits) == len(expected_shapes)
    for i in range(len(splits)):
        assert isinstance(splits[i], tp.Tensor)
        assert isinstance(splits[i].trace_tensor.producer, Split)
        assert splits[i].trace_tensor.rank == len(expected_shapes[i])
        assert splits[i].shape == expected_shapes[i]


class TestSplit:
    def test_basic_instance(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, 2)
        validate_splits(sp, [[2, 5, 6]] * 2)

    def test_empty(self):
        t = tp.Tensor([], dtype=tp.int32)
        sp = tp.split(t, 2)
        validate_splits(sp, [[0]] * 2)

    def test_empty_multiple_dims(self):
        t = tp.Tensor([[], []], dtype=tp.int32)
        sp = tp.split(t, 2)
        validate_splits(sp, [[1, 0]] * 2)

    def test_different_axis(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, 3, dim=2)
        validate_splits(sp, [[4, 5, 2]] * 3)

    def test_index_list(self):
        t = tp.ones((4, 5, 6))
        sp = tp.split(t, [2, 3], dim=0)
        # :2, 2:3, 3:
        expected_shapes = [[2, 5, 6], [1, 5, 6], [1, 5, 6]]
        validate_splits(sp, expected_shapes)

    def test_single_slice(self):
        t = tp.ones((2, 2))
        sp = tp.split(t, 1)
        validate_splits([sp], [[2, 2]])

    @pytest.mark.skip(
        "#203: With no shapes tracking in the frontend it is not possible to catch indivisible split error."
    )
    def test_indivisible_split(self):
        t = tp.ones((2, 2))
        sp = tp.split(t, 3)
        with helper.raises(
            tp.TripyException, match=r"Split input axis 2 must be divisible by the number of sections 3"
        ):
            sp[0].eval()

    def test_indices_out_of_order(self):
        t = tp.ones((5,))
        with helper.raises(
            tp.TripyException, match=r"Split indices must be given in ascending order\, but given \[4, 2, 1\]"
        ):
            sp = tp.split(t, [4, 2, 1])

    def test_empty_indices(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Split indices must not be empty"):
            sp = tp.split(t, [])

    def test_zero_splits(self):
        t = tp.ones((5,))
        with helper.raises(tp.TripyException, match=r"Number of sections argument must be positive, but given 0"):
            sp = tp.split(t, 0)

    def test_invalid_split_dimension(self):
        t = tp.ones(
            (5,),
        )
        with helper.raises(tp.TripyException, match=r"Invalid split dimension 2"):
            sp = tp.split(t, 5, dim=2)
