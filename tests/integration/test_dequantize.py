import cupy as cp
import numpy as np
import pytest

import tripy as tp
from tests import helper
from tests.conftest import skip_if_older_than_sm80, skip_if_older_than_sm89


class TestDequantize:
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_dequantize_int8_per_tensor(self, dtype):
        data = [4, 8]
        input = tp.Tensor(data, dtype=tp.int8)
        scale = 0.5
        scale_tp = tp.Tensor(scale, dtype=dtype)
        dequantized = tp.dequantize(input, scale_tp, dtype)
        # tp.bfloat16 data is converted to np.float32 in cp.from_dlpack().get() API.
        expected = (np.array(data) * scale).astype(np.float32)
        atol = 1e-1 if dtype == tp.bfloat16 else 1e-3
        try:
            output = cp.from_dlpack(dequantized).get()
        except NotImplementedError as e:
            if str(e) == "CuPy does not support bfloat16 yet":
                output = cp.from_dlpack(tp.cast(dequantized, dtype=tp.float32)).get()
            else:
                assert 0 and f"Unsupported output type {dtype}"
        assert np.allclose(output, expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_dequantize_int8_per_channel(self, dtype):
        # TODO: Fix in #153
        if dtype == tp.float16:
            pytest.skip("TRT does not support fp16->int8 per-channel dequant.")
        data = [[4, 8], [4, 8]]
        scale = [0.8, 0.9]
        input = tp.Tensor(data, dtype=tp.int8)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        dequantized = tp.dequantize(input, scale_tp, dtype, dim=0)
        expected = (np.array(data) * np.array(scale).reshape(2, 1)).astype(np.float32)
        atol = 1e-1 if dtype == tp.bfloat16 else 1e-3
        try:
            output = cp.from_dlpack(dequantized).get()
        except NotImplementedError as e:
            if str(e) == "CuPy does not support bfloat16 yet":
                output = cp.from_dlpack(tp.cast(dequantized, dtype=tp.float32)).get()
            else:
                assert 0 and f"Unsupported output type {dtype}"
        assert np.allclose(output, expected, atol=atol)

    # TODO(#161): Update fp8 test to use frontend representation
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_fp8_per_tensor(self, dtype):
        data = [1.0, 1.0]
        input = tp.Tensor(data, dtype=tp.float8)
        scale = 0.5
        scale_tp = tp.Tensor(scale, dtype=dtype)
        dequantized = tp.dequantize(input, scale_tp, dtype)
        assert dequantized.dtype == dtype
        print(dequantized)
        output = cp.from_dlpack(tp.cast(dequantized, tp.float32)).get()
        expected = np.array(data, dtype=np.float32) * scale
        atol = 1e-1 if dtype == tp.bfloat16 else 1e-3
        assert np.allclose(output, expected, atol=atol)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_dequantize_fp8_per_channel(self, dtype):
        data = [[1.0, 1.0], [1.0, 1.0]]
        input = tp.Tensor(data, dtype=tp.float8)
        scale = [0.8, 0.9]
        scale_tp = tp.Tensor(scale, dtype=dtype)
        dequantized = tp.dequantize(input, scale_tp, dtype, dim=0)
        assert dequantized.dtype == dtype
        print(dequantized)
        output = cp.from_dlpack(tp.cast(dequantized, tp.float32)).get()
        expected = (np.array(data) * np.array(scale).reshape(2, 1)).astype(np.float32)
        atol = 1e-1 if dtype == tp.bfloat16 else 1e-3
        assert np.allclose(output, expected, atol=atol)

    def test_negative_non_constant_scale(self):
        data = [[4, 8], [4, 8]]
        input = tp.Tensor(data, dtype=tp.int8)
        scale = tp.ones((2,))
        dequantized = tp.dequantize(input, scale, tp.float32, dim=0)
        with helper.raises(
            tp.TripyException,
            match="Scale must be a constant tensor in dequantize op",
        ):
            print(dequantized)
