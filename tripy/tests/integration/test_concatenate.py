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

import numpy as np
import cupy as cp
import pytest
import nvtripy as tp

from tests import helper


class TestConcatenate:
    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [
            ([(2, 3, 4), (2, 4, 4)], 1),
            ([(2, 3, 4), (2, 3, 2)], -1),
            ([(2, 3, 4), (2, 3, 4)], 0),
            ([(2, 3, 4)], 0),
        ],
    )
    def test_concat(self, tensor_shapes, dim, eager_or_compiled):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        out = eager_or_compiled(tp.concatenate, tensors, dim=dim)
        assert np.array_equal(
            cp.from_dlpack(out).get(), np.concatenate([np.ones(shape) for shape in tensor_shapes], axis=dim)
        )

    @pytest.mark.parametrize(
        "tensor_shapes, dim",
        [
            ([(2, 3, 4), (2, 4, 4)], 0),
            ([(4, 5, 6), (4, 1, 6)], -1),
        ],
    )
    def test_negative_concat(self, tensor_shapes, dim, eager_or_compiled):
        tensors = [tp.ones(shape) for shape in tensor_shapes]
        with helper.raises(tp.TripyException, match=f"all concat input tensors must have the same dimensions"):
            out = eager_or_compiled(tp.concatenate, tensors, dim=dim)
            print(out)
