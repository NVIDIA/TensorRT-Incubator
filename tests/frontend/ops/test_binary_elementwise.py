import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.ops import BinaryElementwise, Comparison

_BINARY_OPS = [
    (BinaryElementwise.Kind.SUM, lambda a, b: a + b),
    (BinaryElementwise.Kind.SUB, lambda a, b: a - b),
    (BinaryElementwise.Kind.POW, lambda a, b: a**b),
    (BinaryElementwise.Kind.MUL, lambda a, b: a * b),
    (BinaryElementwise.Kind.DIV, lambda a, b: a / b),
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
        assert isinstance(out.op, BinaryElementwise)
        if kind in [k for k, _ in _COMPARISON_OPS]:
            assert isinstance(out.op, Comparison)

        if left_side_is_non_tensor and kind in _FLIP_OPS:
            kind = _FLIP_OPS[kind]

        assert out.op.kind == kind

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float16)
        c = a + b

        with helper.raises(
            tp.TripyException,
            # Keep the entire error message here so we'll know if the display becomes horribly corrupted.
            match=r"Incompatible input data types[\.a-zA-Z:|/_0-9-=+,\[\]\s]*? For operation: '\+', data types for all inputs must match, but got: \[float32, float16\].",
            has_stack_info_for=[a, b, c],
        ):
            c.eval()

    def test_invalid_broadcast_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float32)
        c = a + b

        with helper.raises(
            tp.TripyException, match="Input tensors are not broadcast compatible.", has_stack_info_for=[a, b, c]
        ):
            c.eval()
