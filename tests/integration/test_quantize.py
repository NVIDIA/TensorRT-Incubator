import pytest
import numpy as np

import tripy as tp


class TestQuantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_quantize(self, scale, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.int8)
        expected = (np.array(data) / scale).astype(np.int8)
        assert np.array_equal(quantized.numpy(), expected)
