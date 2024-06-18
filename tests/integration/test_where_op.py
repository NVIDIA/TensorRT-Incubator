import cupy as cp
import numpy as np
import pytest

import tripy as tp
from tripy.frontend import Tensor


class TestWhereOp:
    @pytest.mark.parametrize(
        "cond, x, y",
        [
            ((1), (2, 3), (2, 3)),  # Broadcast condition
            ((2), (2, 2), (2, 2)),  # Add extra batch
            ((1,), (1, 3), (1, 3)),
            ((2, 2), (1,), (2,)),  # Broadcast x and y (not equal)
        ],
    )
    def test_where_broadcast_shapes(self, cond, x, y):
        rand_x = np.random.uniform(
            low=0.0,
            high=2.0,
            size=x,
        ).astype(np.float32)
        rand_y = np.random.uniform(low=0.0, high=2.0, size=y).astype(np.float32)
        rand_cond = np.random.uniform(low=0.0, high=2.0, size=cond).astype(np.float32)

        a = Tensor(rand_x)
        b = Tensor(rand_y)
        condition = tp.ones(cond) >= Tensor(rand_cond)
        out = tp.where(condition, a, b)
        assert np.array_equal(
            cp.from_dlpack(out).get(), np.array(np.where((np.ones(cond) >= rand_cond), rand_x, rand_y))
        )

    def test_explicit_condition(self):
        select_indices = tp.Tensor([True, False, True, False], dtype=tp.bool)
        ones = tp.ones((4,), dtype=tp.int32)
        zeros = tp.zeros((4,), dtype=tp.int32)
        w = tp.where(select_indices, ones, zeros)
        assert cp.from_dlpack(w).get().tolist() == [1, 0, 1, 0]
