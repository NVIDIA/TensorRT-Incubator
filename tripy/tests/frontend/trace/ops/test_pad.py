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
from nvtripy.frontend.trace.ops import Pad
from tests import helper


class TestPad:
    def test_op_func(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.pad(a, [(1, 1)])
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Pad)

    def test_infer_rank(self):
        a = tp.Tensor([1, 2, 3, 4])
        a = tp.pad(a, [(1, 1)])
        assert a.trace_tensor.rank == 1

    def test_invalid_pad_length(self):
        with helper.raises(tp.TripyException, match="`pad` length must equal to the rank of `input`."):
            a = tp.Tensor([1, 2, 3, 4])
            a = tp.pad(a, [(1, 1), (1, 1)])

    def test_unsupported_mode(self):
        with helper.raises(tp.TripyException, match="Unsupported padding mode."):
            a = tp.Tensor([1, 2, 3, 4])
            a = tp.pad(a, [(1, 1)], mode="circular")
