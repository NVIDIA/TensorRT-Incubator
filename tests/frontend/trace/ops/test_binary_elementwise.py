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

import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import BinaryElementwise, Comparison

_BINARY_OPS = [
    (BinaryElementwise.Kind.SUM, lambda a, b: a + b),
    (BinaryElementwise.Kind.SUB, lambda a, b: a - b),
    (BinaryElementwise.Kind.POW, lambda a, b: a**b),
    (BinaryElementwise.Kind.MUL, lambda a, b: a * b),
    (BinaryElementwise.Kind.DIV, lambda a, b: a / b),
    (BinaryElementwise.Kind.MAXIMUM, lambda a, b: tp.maximum(a, b)),
    (BinaryElementwise.Kind.MINIMUM, lambda a, b: tp.minimum(a, b)),
]

_COMPARISON_OPS = [
    (Comparison.Kind.LESS, lambda a, b: a < b),
    (Comparison.Kind.LESS_EQUAL, lambda a, b: a <= b),
    (Comparison.Kind.EQUAL, lambda a, b: a == b),
    (Comparison.Kind.NOT_EQUAL, lambda a, b: a != b),
    (Comparison.Kind.GREATER_EQUAL, lambda a, b: a >= b),
    (Comparison.Kind.GREATER, lambda a, b: a > b),
]

# Ops that are flipped instead of calling a right-side version.
_FLIP_OPS = {}
for key, val in {
    Comparison.Kind.LESS: Comparison.Kind.GREATER,
    Comparison.Kind.LESS_EQUAL: Comparison.Kind.GREATER_EQUAL,
}.items():
    _FLIP_OPS[key] = val
    _FLIP_OPS[val] = key


class TestBinaryElementwise:
    # Make sure that we can support non-tensor arguments as either lhs or rhs.
    # Comparison operators have no right-side overload - instead, they will simply
    # call their opposite.
    @pytest.mark.parametrize(
        "lhs, rhs, left_side_is_non_tensor",
        [
            (tp.Tensor([1.0]), tp.Tensor([2.0]), False),
            (tp.Tensor([1.0]), np.array([2.0], dtype=np.float32), False),
            # shape of (0,) is broadcastable with (1,)
            (tp.Tensor([], dtype=tp.float32), tp.Tensor([1.0], dtype=tp.float32), False),
            (np.array([1.0], dtype=np.float32), tp.Tensor([2.0]), True),
            (tp.Tensor([1.0]), 2.0, False),
            (1.0, tp.Tensor([2.0]), True),
        ],
        ids=lambda obj: type(obj).__qualname__,
    )
    @pytest.mark.parametrize("kind, func", _BINARY_OPS + _COMPARISON_OPS)
    def test_op_funcs(self, kind, func, lhs, rhs, left_side_is_non_tensor):
        out = func(lhs, rhs)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.trace_tensor.producer, BinaryElementwise)
        if kind in [k for k, _ in _COMPARISON_OPS]:
            assert isinstance(out.trace_tensor.producer, Comparison)

        if left_side_is_non_tensor and kind in _FLIP_OPS:
            kind = _FLIP_OPS[kind]

        assert out.trace_tensor.producer.kind == kind
        assert out.trace_tensor.rank == 1

    @pytest.mark.parametrize(
        "lhs, rhs, expected_rank",
        [
            (tp.Tensor([1.0]), tp.Tensor([2.0]), 1),
            (tp.Tensor([1.0]), np.array([2.0], dtype=np.float32), 1),
            (np.array([1.0], dtype=np.float32), tp.Tensor([2.0]), 1),
            (tp.Tensor([1.0]), 2.0, 1),
            (1.0, tp.Tensor([2.0]), 1),
            (tp.ones((2, 3)), 2.0, 2),
        ],
    )
    def test_infer_rank(self, lhs, rhs, expected_rank):
        out = lhs + rhs
        assert out.trace_tensor.rank == expected_rank

    def test_mismatched_dtypes_fails(self):
        a = tp.Tensor([1, 2], dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)

        with helper.raises(
            tp.TripyException,
            # Keep the entire error message here so we'll know if the display becomes horribly corrupted.
            match=r"For operation: '\+', data types for all inputs must match, but got: \[float32, float16\].",
            has_stack_info_for=[a, b],
        ):
            c = a + b

    def test_invalid_broadcast_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float32)
        c = a + b

        with helper.raises(
            tp.TripyException,
            match=r"size of operand dimension 1 \(3\) is not compatible with size of result dimension",
            has_stack_info_for=[a, b, c],
        ):
            c.eval()
