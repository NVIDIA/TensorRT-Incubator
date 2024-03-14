import pytest
import numpy as np

import tripy as tp


class TestDequantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_dequantize(self, scale, dtype):
        data = [4, 8]
        input = tp.Tensor(data, dtype=tp.int8)
        dequantized = tp.dequantize(input, scale, dtype)
        expected = (np.array(data) * scale).astype(dtype.name)
        assert np.array_equal(dequantized.numpy(), expected)
