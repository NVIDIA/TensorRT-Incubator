import pytest
import numpy as np

import tripy as tp


class TestQuantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_quantize_int8_per_tensor(self, scale, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.int8)
        expected = (np.array(data) / scale).astype(np.int8)
        assert np.array_equal(quantized.numpy(), expected)

    @pytest.mark.parametrize("scale", [[0.2, 0.1], [0.5, 0.5]])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_quantize_int8_per_channel(self, scale, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.int8, dim=0)
        expected = (np.array(data) / np.array(scale).reshape(2, 1)).astype(np.int8)
        assert np.array_equal(quantized.numpy(), expected)

    # TODO(#161): Update fp8 test to check output value
    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_quantize_fp8_per_tensor(self, scale, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.float8)
        assert quantized.dtype == tp.float8
        assert quantized.numpy().dtype == np.uint8

    @pytest.mark.parametrize("scale", [[0.2, 0.1], [0.5, 0.5]])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_quantize_fp8_per_channel(self, scale, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.float8, dim=0)
        assert quantized.dtype == tp.float8
        assert quantized.numpy().dtype == np.uint8
