import numpy as np
import pytest

from tripy.frontend import Tensor
from tripy.frontend.ops import ones, where


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
        condition = ones(cond) >= Tensor(rand_cond)
        out = where(condition, a, b)
        assert np.array_equal(out.numpy(), np.array(np.where((np.ones(cond) >= rand_cond), rand_x, rand_y)))
