import numpy as np
import pytest
from textwrap import dedent
import tripy as tp
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

        with pytest.raises(
            tp.TripyException,
            # Keep the entire error message here so we'll know if the display becomes horribly corrupted.
            match=dedent(
                rf"""
                Incompatible input data types.
                    For expression:

                    | {__file__}:[0-9]+
                    | -------------------------------------------------------
                    |         c = a \+ b

                    For operation: '\+', data types for all inputs must match, but got: \[float32, float16\].

                    Input 0 was:

                    | /tripy/tripy/[a-z/_\.]+:[0-9]+
                    | ---------------------------------------------------
                    |     return full\(shape, 1, dtype\)

                    Called from:

                    | /tripy/tests/frontend/ops/test_binary_elementwise.py:64
                    | -------------------------------------------------------
                    |         a = tp.ones\(\(2, 3\), dtype=tp.float32\)

                    Input 1 was:

                    | /tripy/tripy/[a-z/_\.]+:32
                    | ---------------------------------------------------
                    |     return full\(shape, 1, dtype\)

                    Called from:

                    | {__file__}:[0-9]+
                    | -------------------------------------------------------
                    |         b = tp.ones\(\(2, 3\), dtype=tp.float16\)
                """
                # """,
            ).strip(),
        ) as exc:
            c.eval()
        print(str(exc.value))

    def test_invalid_broadcast_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float32)
        c = a + b

        with pytest.raises(tp.TripyException, match="Input tensors are not broadcast compatible.") as exc:
            c.eval()
        print(str(exc.value))
