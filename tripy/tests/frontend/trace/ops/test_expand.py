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


import nvtripy as tp
from tests import helper
from nvtripy.frontend.trace.ops import Expand


class TestExpand:
    def test_func_op(self):
        a = tp.ones((2, 1))
        a = tp.expand(a, (2, 2))
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Expand)

    def test_invalid_small_size(self):
        a = tp.ones((2, 1, 1))

        with helper.raises(
            tp.TripyException,
            match="The length of `sizes` must be greater or equal to input tensor's rank.",
        ):
            b = tp.expand(a, (2, 2))

    def test_invalid_prepended_dim(self):
        # We cannot use -1 if we are prepending a new dimension.
        a = tp.ones((2,))

        with helper.raises(tp.TripyException, match="Cannot use -1 for prepended dimension."):
            b = tp.expand(a, (-1, 2))

    def test_invalid_mismatch_size(self):
        a = tp.ones((2, 1))
        b = tp.expand(a, (4, 2))

        with helper.raises(
            tp.TripyException,
            match=r"size of operand dimension 0 \(2\) is not compatible with size of result dimension 0 \(4\)",
            has_stack_info_for=[a, b],
        ):
            b.eval()

    def test_infer_rank(self):
        a = tp.ones((2, 1))
        a = tp.expand(a, (2, 2))
        assert a.trace_tensor.rank == 2
