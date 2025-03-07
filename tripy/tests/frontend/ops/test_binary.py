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


class TestBinaryElementwise:
    # Make sure that we can support non-tensor arguments as either lhs or rhs.
    # Comparison operators have no right-side overload - instead, they will simply
    # call their opposite.
    @pytest.mark.parametrize(
        "lhs, rhs",
        [
            (tp.Tensor([1.0]), tp.Tensor([2.0])),
            # shape of (0,) is broadcastable with (1,)
            (tp.Tensor([], dtype=tp.float32), tp.Tensor([1.0], dtype=tp.float32)),
            (tp.Tensor([1.0]), 2.0),
            (1.0, tp.Tensor([2.0])),
        ],
        ids=lambda obj: type(obj).__qualname__,
    )
    @pytest.mark.parametrize(
        "func",
        [
            (lambda a, b: a + b),
            (lambda a, b: a - b),
            (lambda a, b: a**b),
            (lambda a, b: a * b),
            (lambda a, b: a / b),
            (lambda a, b: a // b),
            (lambda a, b: a < b),
            (lambda a, b: a <= b),
            (lambda a, b: a == b),
            (lambda a, b: a != b),
            (lambda a, b: a >= b),
            (lambda a, b: a > b),
        ],
    )
    def test_non_tensor_types(self, func, lhs, rhs):
        out = func(lhs, rhs)
        assert isinstance(out, tp.Tensor)

    def test_mismatched_dtypes_fails(self):
        a = tp.Tensor([1, 2], dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)

        with helper.raises(
            tp.TripyException,
            # Keep the entire error message here so we'll know if the display becomes horribly corrupted.
            match=r"Mismatched data types in '__add__'.",
            has_stack_info_for=[a, b],
        ):
            c = a + b

    def test_invalid_broadcast_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float32)
        c = a + b

        with helper.raises(tp.TripyException, match=r"broadcast dimensions must be conformable"):
            c.eval()

    def test_dimension_size_inputs(self):
        d = tp.DimensionSize(1)

        # Operations on only DimensionSizes will yield a DimensionSize
        out = d + d
        assert isinstance(out, tp.DimensionSize)

        # Otherwise, a Tensor is yielded.
        a = tp.Tensor([1, 2])
        out = a + d
        assert isinstance(out, tp.Tensor) and not isinstance(out, tp.DimensionSize)
