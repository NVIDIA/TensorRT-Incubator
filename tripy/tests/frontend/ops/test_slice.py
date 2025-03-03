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

import cupy as cp
import nvtripy as tp
from nvtripy.trace.ops.slice import Slice
from tests import helper


class TestSlice:
    def test_slice_of_inline_output(self):
        a = tp.Tensor([1, 2, 3, 4])
        # The start and stop params use clamp bound, but the step parameter doesn't.
        s = (a + a)[3:4:]
        assert isinstance(s, tp.Tensor)
        assert isinstance(s.trace_tensor.producer, Slice)

        # input 0 is a + a, so it's not one of the slice params
        slice_inputs = s.trace_tensor.producer.inputs[1:]
        assert len(slice_inputs) == 3

        assert any(frame.function == "clamp_bound" for frame in slice_inputs[0].stack_info)
        assert any(frame.function == "clamp_bound" for frame in slice_inputs[1].stack_info)
        assert not any(frame.function == "clamp_bound" for frame in slice_inputs[2].stack_info)

    def test_incorrect_index_size(self):
        with helper.raises(
            tp.TripyException,
            match=r"Input tensor has a rank of 2 but was attempted to be sliced with 3 indices",
        ):
            a = tp.Tensor([[1, 2], [3, 4]])
            b = a[:, :, 0:1]
            b.eval()

    def test_invalid_index(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(tp.TripyException, match="out of bounds access", has_stack_info_for=[a]):
            a[3].eval()

    def test_invalid_multiple_dims(self):
        a = tp.ones((2, 3, 4))
        with helper.raises(tp.TripyException, match="out of bounds access", has_stack_info_for=[a]):
            a[5, 3].eval()
