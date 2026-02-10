#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class TestMatMul:
    def test_0d_matrix_fails(self):
        a = tp.ones(tuple(), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)

        with helper.raises(tp.TripyException, match="Input tensors must have at least 1 dimension."):
            c = a @ b
            c.eval()

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((3, 2), dtype=tp.float16)

        with helper.raises(tp.TripyException, match="Invalid inputs for function: '__matmul__'."):
            c = a @ b

    def test_incompatible_1d_shapes_fails(self):
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = a @ b

        with helper.raises(
            tp.TripyException, match="last dimension of input0 = 2 and last dimension of input1 = 3 but must match"
        ):
            c.eval()

    def test_incompatible_2d_shapes_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((3, 6), dtype=tp.float32)
        c = a @ b

        with helper.raises(
            tp.TripyException, match="last dimension of input0 = 4 and second to last dimension of input1 = 3"
        ):
            c.eval()
