import cupy as cp
import numpy as np
import pytest
import torch

import tripy as tp
from tests import helper
from tests.conftest import skip_if_older_than_sm80, skip_if_older_than_sm89


class TestQuantize:
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_tensor(self, dtype):
        data = [1.0, 2.0]
        scale = 0.5
        input = tp.Tensor(data, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input, scale_tp, tp.int8)
        expected = (np.array(data) / scale).astype(np.int8)
        assert np.array_equal(cp.from_dlpack(quantized).get(), expected)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_channel(self, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        scale = [0.2, 0.1]
        input = tp.Tensor(data, dtype=dtype)
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input, scale_tp, tp.int8, dim=0)
        expected = (np.array(data) / np.array(scale).reshape(2, 1)).astype(np.int8)
        assert np.array_equal(cp.from_dlpack(quantized).get(), expected)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_tensor(self, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        scale = 0.5
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input, scale_tp, tp.float8)
        assert quantized.dtype == tp.float8
        output = cp.from_dlpack(tp.cast(quantized, dtype=tp.float32)).get()
        expected = np.array(data) / np.array(scale)
        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_channel(self, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        input = tp.Tensor(data, dtype=dtype)
        scale = [0.2, 0.1]
        scale_tp = tp.Tensor(scale, dtype=dtype)
        quantized = tp.quantize(input, scale_tp, tp.float8, dim=0)
        assert quantized.dtype == tp.float8
        output = cp.from_dlpack(tp.cast(quantized, dtype=tp.float32)).get()
        expected = np.array(data) / np.array(scale).reshape(2, 1)
        assert np.array_equal(output, expected)

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @pytest.mark.parametrize("quant_mode", ["block-wise", "per-tensor", "per-channel-0", "per-channel-1"])
    def test_qdq_int4(self, dtype, quant_mode):
        TORCH_DTYPES = {
            tp.float32: torch.float32,
            tp.float16: torch.float16,
            tp.bfloat16: torch.bfloat16,
        }
        if quant_mode == "block-wise":
            dim = None
            scale = torch.ones((2, 4), dtype=TORCH_DTYPES[dtype])
        elif quant_mode == "per-tensor":
            dim = None
            scale = torch.tensor(1.0, dtype=TORCH_DTYPES[dtype])
        elif quant_mode.endswith("0"):
            dim = 0
            scale = torch.ones((4,), dtype=TORCH_DTYPES[dtype])
        elif quant_mode.endswith("1"):
            dim = 1
            scale = torch.ones((4,), dtype=TORCH_DTYPES[dtype])

        data = tp.ones((4, 4), dtype=dtype)
        scale = tp.Tensor(scale)
        quantized = tp.quantize(data, scale, tp.int4, dim)
        dequantized = tp.dequantize(quantized, scale, dtype, dim)
        try:
            dq_out = cp.from_dlpack(dequantized).get()
        except NotImplementedError as e:
            if str(e) == "CuPy does not support bfloat16 yet":
                dq_out = cp.from_dlpack(tp.cast(dequantized, dtype=tp.float32)).get()
            else:
                assert 0 and f"Unsupported output type {dtype}"
        try:
            data_out = cp.from_dlpack(data).get()
        except NotImplementedError as e:
            if str(e) == "CuPy does not support bfloat16 yet":
                data_out = cp.from_dlpack(tp.cast(data, dtype=tp.float32)).get()
            else:
                assert 0 and f"Unsupported output type {dtype}"
        assert np.array_equal(dq_out, data_out)

    def test_negative_non_constant_scale(self):
        input = tp.ones((4, 4))
        scale = tp.ones((4,))
        quantized = tp.quantize(input, scale, tp.int8, dim=0)
        with helper.raises(
            tp.TripyException,
            match="Scale must be a constant tensor in quantize op",
        ):
            print(quantized)
