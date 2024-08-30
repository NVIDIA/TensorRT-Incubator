import tripy as tp

import pytest
import numpy as np


class TestRepeat:
    @pytest.mark.parametrize(
        "repeats,dim",
        [
            (1, 0),
            (2, 0),
            (2, -1),
            (2, 1),
            (0, 1),
        ],
    )
    def test_repeat(self, repeats, dim):
        inp = np.arange(4, dtype=np.int32).reshape((2, 2))

        out = tp.repeat(tp.Tensor(inp), repeats, dim)
        expected = np.repeat(inp, repeats, dim)

        assert np.array_equal(np.from_dlpack(tp.copy(out, device=tp.device("cpu"))), expected)
