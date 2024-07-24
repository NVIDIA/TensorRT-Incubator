import cupy as cp
import numpy as np

import tripy as tp


class TestFull:
    def test_normal_shape(self):
        out = tp.full((2, 2), 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 2), 5.0, np.float32))

    def test_shape_tensor(self):
        a = tp.ones((2, 3))
        out = tp.full(a.shape, 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 3), 5.0, np.float32))

    def test_mixed_shape(self):
        a = tp.ones((2, 3))
        out = tp.full((a.shape[0], 4), 5.0, tp.float32)
        assert np.array_equal(cp.from_dlpack(out).get(), np.full((2, 4), 5.0, np.float32))
