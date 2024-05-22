import cupy as cp
import numpy as np

import tripy as tp


class TestStronglyTyped:
    """
    Sanity test that strongly typed mode is enabled.
    """

    def test_fp16_no_overflow(self):
        a = tp.Tensor([10000, 60000], dtype=tp.float32)
        a = tp.sum(a)  # 7e+4 is out of fp16 upperbound
        a = a / 5.0
        a = tp.cast(a, tp.float16)

        assert cp.from_dlpack(a).get() == np.array([14000], dtype=np.float16)
