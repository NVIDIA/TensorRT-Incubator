import pytest
import re
import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Where


class TestWhere:
    def test_op_func(self):
        a = tp.Tensor([[0, 1], [0, 1]], shape=(2, 2))
        b = tp.Tensor([[0, 0], [1, 1]], shape=(2, 2))
        condition = a >= b
        a = tp.where(condition, a, b)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Where)

    def test_bool_condition(self):
        cond = tp.Tensor([False, True, False], dtype=tp.bool)
        a = tp.Tensor([1, 2, 3], shape=(3,), dtype=tp.int32)
        b = tp.Tensor([4, 5, 6], shape=(3,), dtype=tp.int32)
        w = tp.where(cond, a, b)
        assert isinstance(w, tp.Tensor)
        assert isinstance(w.trace_tensor.producer, Where)

    @pytest.mark.skip("Test segfaults due to https://gitlab-master.nvidia.com/initialdl/mlir-tensorrt/-/issues/885")
    def test_mismatched_input_shapes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((3,), dtype=tp.float32)
        c = tp.where(cond, a, b)

        with helper.raises(
            tp.TripyException,
            match=re.escape("size of operand dimension 0 (2) is not compatible with size of result dimension 0 (3)"),
            has_stack_info_for=[a, b, c, cond],
        ):
            c.eval()

    def test_mismatched_input_dtypes(self):
        cond = tp.ones((2,), dtype=tp.float32) > tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float16)

        with helper.raises(tp.TripyException, match="Incompatible input data types.", has_stack_info_for=[a, b, cond]):
            c = tp.where(cond, a, b)

    def test_condition_is_not_bool(self):
        cond = tp.ones((2,), dtype=tp.float32)
        a = tp.ones((2,), dtype=tp.float32)
        b = tp.ones((2,), dtype=tp.float32)

        with helper.raises(
            tp.TripyException, match="Condition input must have boolean type.", has_stack_info_for=[a, b, cond]
        ):
            c = tp.where(cond, a, b)

    def test_infer_rank(self):
        a = tp.Tensor([[0, 1], [0, 1]], shape=(2, 2))
        b = tp.Tensor([[0, 0], [1, 1]], shape=(2, 2))
        condition = a >= b
        a = tp.where(condition, a, b)
        assert a.trace_tensor.rank == 2


class TestMaskedFill:
    def test_op_func(self):
        a = tp.Tensor([0, 1, 0, 1])
        b = tp.Tensor([0, 0, 1, 0])
        mask = a == b
        a = tp.masked_fill(a, mask, -1)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Where)

    def test_condition_is_not_bool(self):
        a = tp.Tensor([0, 1, 0, 1])
        mask = tp.Tensor([1.0, 2.0, 3.0, 4.0])

        with helper.raises(
            tp.TripyException, match="Condition input must have boolean type.", has_stack_info_for=[a, mask]
        ):
            b = tp.masked_fill(a, mask, -1)

    def test_infer_rank(self):
        a = tp.Tensor([0, 1, 0, 1])
        b = tp.Tensor([0, 0, 1, 0])
        mask = a == b
        a = tp.masked_fill(a, mask, -1)
        assert a.trace_tensor.rank == 1
