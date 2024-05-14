import numpy as np
import pytest

import tripy as tp
from tests.conftest import skip_if_older_than_sm89, skip_if_older_than_sm80


class TestDequantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_dequantize_int8_per_tensor(self, scale, dtype):
        data = [4, 8]
        input = tp.Tensor(data, dtype=tp.int8)
        dequantized = tp.dequantize(input, scale, dtype)
        expected = (np.array(data) * scale).astype(dtype.name)
        assert np.array_equal(dequantized.numpy(), expected)

    @pytest.mark.parametrize("scale", [[0.8, 0.9], [0.5, 0.5]])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
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
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_fp8_per_tensor(self, scale, dtype):
        data = [1.0, 1.0]
        input = tp.Tensor(data, dtype=tp.float8)
        dequantized = tp.dequantize(input, scale, dtype)
        assert dequantized.dtype == dtype
        print(dequantized)

    @pytest.mark.parametrize("scale", [[0.8, 0.9], [0.5, 0.5]])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_fp8_per_channel(self, scale, dtype):
        data = [[1.0, 1.0], [1.0, 1.0]]
        input = tp.Tensor(data, dtype=tp.float8)
        dequantized = tp.dequantize(input, scale, dtype, dim=0)
        assert dequantized.dtype == dtype
        print(dequantized)
