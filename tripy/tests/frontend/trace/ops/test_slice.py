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

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Slice


class TestSlice:
    def test_op_func_all_partial(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = a[:2]
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Slice)

    def test_incorrect_index_size(self):
        with helper.raises(
            tp.TripyException,
            match=r"Input tensor has a rank of 2 but was attempted to be sliced with 3 indices",
        ) as exc:
            a = tp.Tensor([[1, 2], [3, 4]])
            b = a[:, :, 0:1]
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((2, 3))
        a = a[:2, :]
        assert a.trace_tensor.rank == 2

    def test_scalar_index(self):
        a = tp.ones((2, 3, 4))
        assert a[0].shape == [3, 4]
        assert a[0:1].shape == [1, 3, 4]
        assert cp.from_dlpack(a[0].shape).get().tolist() == [3, 4]

    def test_end_clamping(self):
        a = tp.ones((2, 3, 4))
        # equivalent to a[0:2, 0:3, 3:4]
        assert a[0:7, 0:12, 3:5].shape == [2, 3, 1]

    def test_tensor_index(self):
        idx = tp.Tensor(1, dtype=tp.int32)
        a = tp.ones((2, 3))
        b = a[idx]
        assert b.shape == [3]

    def test_empty_slice(self):
        a = tp.ones((2, 3, 4))
        b = a[3:2:1]
        assert b.shape == [0, 3, 4]
        assert cp.from_dlpack(b).get().tolist() == []

    def test_invalid_index(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(
            tp.TripyException,
            # note that the stack trace includes an ANSI color code before the caret
            # Looks like:
            # |             a[3].eval()
            # |               ^
            match=r"\| {13}a\[3\]\.eval\(\)\n\s*\| {15}\x1b\[38;5;1m\^",
            has_stack_info_for=[a],
        ):
            a[3].eval()

    def test_invalid_multiple_dims(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(
            tp.TripyException,
            match="limit index 6 is larger than dimension size 2 in dimension 0",
        ):
            a[5, 3].eval()
