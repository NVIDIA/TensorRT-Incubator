import tripy as tp
from tests import helper
from tripy.frontend.trace.ops.where import Where


class TestWhere:
    def test_op_func(self):
        a = tp.Tensor([[0, 1], [0, 1]], shape=(2, 2))
        b = tp.Tensor([[0, 0], [1, 1]], shape=(2, 2))
        condition = a >= b
        a = tp.where(condition, a, b)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Where)

    def test_mismatched_input_shapes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with helper.raises(
            tp.TripyException, match="Input tensors are not broadcast compatible.", has_stack_info_for=[a, b, c, cond]
        ):
            c.eval()

    def test_mismatched_input_dtypes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)
        c = tp.where(cond, a, b)

        with helper.raises(
            tp.TripyException, match="Incompatible input data types.", has_stack_info_for=[a, b, c, cond]
        ):
            c.eval()

    def test_condition_is_not_bool(self):
        cond = tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with helper.raises(
            tp.TripyException, match="Condition input must have boolean type.", has_stack_info_for=[a, b, c, cond]
        ):
            c.eval()


class TestMaskedFill:
    def test_op_func(self):
        a = tp.Tensor([0, 1, 0, 1])
        b = tp.Tensor([0, 0, 1, 0])
        mask = a == b
        a = a.masked_fill(mask, -1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.op, Where)

    def test_condition_is_not_bool(self):
        a = tp.Tensor([0, 1, 0, 1])
        mask = tp.Tensor([1.0, 2.0, 3.0, 4.0])
        b = a.masked_fill(mask, -1)

        with helper.raises(
            tp.TripyException, match="Condition input must have boolean type.", has_stack_info_for=[a, mask, b]
        ):
            b.eval()
