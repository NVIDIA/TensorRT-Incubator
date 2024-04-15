import pytest
import numpy as np

import tripy as tp


class TestDequantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_dequantize_int8_per_tensor(self, scale, dtype):
        data = [4, 8]
        input = tp.Tensor(data, dtype=tp.int8)
        dequantized = tp.dequantize(input, scale, dtype)
        expected = (np.array(data) * scale).astype(dtype.name)
        assert np.array_equal(dequantized.numpy(), expected)

    @pytest.mark.parametrize("scale", [[0.8, 0.9], [0.5, 0.5]])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_dequantize_int8_per_channel(self, scale, dtype):
        # TODO: Fix in #153
        if dtype == tp.float16:
            pytest.skip("TRT does not support fp16->int8 per-channel dequant.")
        data = [[4, 8], [4, 8]]
        input = tp.Tensor(data, dtype=tp.int8)
        dequantized = tp.dequantize(input, scale, dtype, dim=0)
        expected = (np.array(data) * np.array(scale).reshape(2, 1)).astype(dtype.name)
        assert np.array_equal(dequantized.numpy(), expected)

    # TODO(#161): Update fp8 test to use frontend representation
    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_dequantize_fp8_per_tensor(self, scale, dtype):
        data = [56, 56]  # uint8 data for fp8 [1.0, 1.0]
        input = tp.Tensor(data, dtype=tp.float8)
        dequantized = tp.dequantize(input, scale, dtype)
        assert dequantized.dtype == dtype
        print(dequantized)

    @pytest.mark.parametrize("scale", [[0.8, 0.9], [0.5, 0.5]])
    @pytest.mark.parametrize("dtype", [tp.float32, tp.float16])
    def test_dequantize_fp8_per_channel(self, scale, dtype):
        data = [[56, 56], [56, 56]]
        input = tp.Tensor(data, dtype=tp.float8)
        dequantized = tp.dequantize(input, scale, dtype, dim=0)
        assert dequantized.dtype == dtype
        print(dequantized)
