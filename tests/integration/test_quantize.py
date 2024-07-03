import cupy as cp
import numpy as np
import pytest

import tripy as tp
from tests import helper
from tests.conftest import skip_if_older_than_sm80, skip_if_older_than_sm89

import mlir_tensorrt.runtime


@pytest.mark.skip("https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/issues/219")
class TestQuantize:

    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_tensor(self, scale, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.int8)
        expected = (np.array(data) / scale).astype(np.int8)
        assert np.array_equal(cp.from_dlpack(quantized).get(), expected)

    @pytest.mark.parametrize("scale", [[0.2, 0.1], [0.5, 0.5]])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    def test_quantize_int8_per_channel(self, scale, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.int8, dim=0)
        expected = (np.array(data) / np.array(scale).reshape(2, 1)).astype(np.int8)
        assert np.array_equal(cp.from_dlpack(quantized).get(), expected)

    # TODO(#161): Update fp8 test to check output value
    @pytest.mark.parametrize("scale", [0.5, 0.9])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_tensor(self, scale, dtype):
        data = [1.0, 2.0]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.float8)
        assert quantized.dtype == tp.float8
        try:
            output = cp.from_dlpack(quantized).get()
            assert (
                0
                and f"As of 5/21/2024 Fp8 output type is not supported. Remove exception below if this line is logged."
            )
        except mlir_tensorrt.runtime._mlir_libs._api.MTRTException as e:
            # tp.float8 data must be converted to np.float32.
            if (
                str(e)
                == "InvalidArgument: InvalidArgument: Scalar type code [f8e4m3fn] conversion to DLPackDataTypeCode is not supported."
            ):
                output = cp.from_dlpack(tp.cast(quantized, dtype=tp.float32)).get()
                assert output.dtype == np.float32
            else:
                assert 0 and f"Unsupported output type {dtype}"

    @pytest.mark.parametrize("scale", [[0.2, 0.1], [0.5, 0.5]])
    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @skip_if_older_than_sm89
    def test_quantize_fp8_per_channel(self, scale, dtype):
        data = [[1.0, 2.0], [3.0, 4.0]]
        input = tp.Tensor(data, dtype=dtype)
        quantized = tp.quantize(input, scale, tp.float8, dim=0)
        assert quantized.dtype == tp.float8
        try:
            output = cp.from_dlpack(quantized).get()
            assert (
                0
                and f"As of 5/21/2024 Fp8 output type is not supported. Remove exception below if this line is logged."
            )
        except mlir_tensorrt.runtime._mlir_libs._api.MTRTException as e:
            # tp.float8 data must be converted to np.float32.
            if (
                str(e)
                == "InvalidArgument: InvalidArgument: Scalar type code [f8e4m3fn] conversion to DLPackDataTypeCode is not supported."
            ):
                output = cp.from_dlpack(tp.cast(quantized, dtype=tp.float32)).get()
                assert output.dtype == np.float32
            else:
                assert 0 and f"Unsupported output type {dtype}"

    @pytest.mark.parametrize(
        "dtype", [tp.float32, tp.float16, pytest.param(tp.bfloat16, marks=skip_if_older_than_sm80)]
    )
    @pytest.mark.parametrize("quant_mode", ["block-wise", "per-tensor", "per-channel-0", "per-channel-1"])
    def test_qdq_int4(self, dtype, quant_mode):
        if quant_mode == "block-wise":
            dim = None
            scale = tp.ones((2, 4))
        elif quant_mode == "per-tensor":
            dim = None
            scale = 1.0
        elif quant_mode.endswith("0"):
            dim = 0
            scale = tp.ones((4,))
        elif quant_mode.endswith("1"):
            dim = 1
            scale = tp.ones((4,))

        data = tp.ones((4, 4), dtype=dtype)
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

    @pytest.mark.parametrize("quant_mode", ["block-wise", "per-tensor"])
    def test_negative_invalid_dim(self, quant_mode):
        if quant_mode == "block-wise":
            dim = 0
            scale = tp.ones((2, 4))
        elif quant_mode == "per-tensor":
            dim = 0
            scale = 1.0

        data = tp.ones((2, 4))
        quantized = tp.quantize(data, scale, tp.int4, dim=dim)
        with helper.raises(
            tp.TripyException,
            match="'tensorrt.quantize' op if axis is provided, scale must be a 1D tensor for per channel quantization",
        ):
            print(quantized)

    def test_negative_per_channel_scale_size_mismatch(self):
        data = tp.ones((2, 4))
        scale = [1.0] * 4
        quantized = tp.quantize(data, scale, tp.int8, dim=0)
        with helper.raises(
            tp.TripyException,
            match="'tensorrt.quantize' op expected the scales size to match the quantization axis of input tensor",
        ):
            print(quantized)

    def test_negative_blockwise_invalid_dtype(self):
        data = tp.ones((4, 4))
        scale = tp.ones((2, 4))
        quantized = tp.quantize(data, scale, tp.int8)
        with helper.raises(
            tp.TripyException,
            match="'tensorrt.quantize' op 2D scale is supported only for quantizing INT4 output",
        ):
            print(quantized)
