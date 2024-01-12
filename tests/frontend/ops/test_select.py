import pytest

import tripy as tp
from tripy.frontend.ops.select import Select


class TestWhere:
    def test_op_func(self):
        a = tp.Tensor([[0, 1], [0, 1]])
        b = tp.Tensor([[0, 0], [1, 1]])
        condition = a >= b
        a = tp.where(condition, a, b)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Select)

    def test_mismatched_input_shapes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with pytest.raises(tp.TripyException, match="Incompatible input shapes.") as exc:
            c.eval()
        print(str(exc.value))

    def test_mismatched_input_dtypes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)
        c = tp.where(cond, a, b)

        with pytest.raises(tp.TripyException, match="Incompatible input data types.") as exc:
            c.eval()
        print(str(exc.value))

    def test_condition_is_not_bool(self):
        cond = tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with pytest.raises(tp.TripyException, match="Condition input must have boolean type.") as exc:
            c.eval()
        print(str(exc.value))


class TestMaskedFill:
    def test_op_func(self):
        a = tp.Tensor([0, 1, 0, 1])
        b = tp.Tensor([0, 0, 1, 0])
        mask = a == b
        a = a.masked_fill(mask, -1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Select)

    def test_condition_is_not_bool(self):
        a = tp.Tensor([0, 1, 0, 1])
        mask = tp.Tensor([1.0, 2.0, 3.0, 4.0])
        a = a.masked_fill(mask, -1)

        with pytest.raises(tp.TripyException, match="Condition input must have boolean type.") as exc:
            a.eval()
        print(str(exc.value))
