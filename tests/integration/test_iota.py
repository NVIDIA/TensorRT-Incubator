import numpy as np

import tripy as tp


class TestIota:
    def test_multi_dimensional(self):
        output = tp.iota([2, 3], dim=1)
        expected = np.broadcast_to(np.arange(0, 3, dtype=np.float32), (2, 3))

        assert np.array_equal(output.numpy(), expected)
