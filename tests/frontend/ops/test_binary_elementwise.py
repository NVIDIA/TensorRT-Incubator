import pytest

import tripy as tp
from tripy.frontend.ops import BinaryElementwise


_BINARY_OPS = {
    BinaryElementwise.Kind.SUM: lambda a, b: a + b,
    BinaryElementwise.Kind.LESS: lambda a, b: a < b,
    BinaryElementwise.Kind.LESS_EQUAL: lambda a, b: a <= b,
    BinaryElementwise.Kind.EQUAL: lambda a, b: a == b,
    BinaryElementwise.Kind.NOT_EQUAL: lambda a, b: a != b,
    BinaryElementwise.Kind.GREATER_EQUAL: lambda a, b: a >= b,
    BinaryElementwise.Kind.GREATER: lambda a, b: a > b,
}


class TestBinaryElementwise:
    @pytest.mark.parametrize("func, kind", [(func, kind) for kind, func in _BINARY_OPS.items()])
    def test_op_funcs(self, func, kind):
        a = tp.Tensor([1])
        b = tp.Tensor([2])

        out = func(a, b)
        assert isinstance(out, tp.Tensor)
        assert isinstance(out.op, BinaryElementwise)
        assert out.op.kind == kind

    def test_mismatched_dtypes_fails(self):
        a = tp.ones((2, 3), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float16)
        c = a + b

        with pytest.raises(tp.TripyException, match="Incompatible input data types.") as exc:
            c.eval()
        print(str(exc.value))

    def test_mismatched_ranks_fails(self):
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float16)
        c = a + b

        with pytest.raises(tp.TripyException, match="Incompatible input tensor ranks.") as exc:
            c.eval()
        print(str(exc.value))

    def test_invalid_broadcast_fails(self):
        a = tp.ones((2, 4), dtype=tp.float32)
        b = tp.ones((2, 3), dtype=tp.float16)
        c = a + b

        with pytest.raises(tp.TripyException, match="Input tensors are not broadcast compatible.") as exc:
            c.eval()
        print(str(exc.value))
