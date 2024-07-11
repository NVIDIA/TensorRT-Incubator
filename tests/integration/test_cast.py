import cupy as cp
import numpy as np
import pytest
import torch

import tripy as tp
from tests.conftest import skip_if_older_than_sm89


class TestCast:
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize(
        "input_dtype, target_dtype",
        [
            (np.int32, np.float32),
            (np.float32, np.int32),
            (np.int64, np.float32),
            (np.float32, np.int64),
            (np.float32, np.float64),
            (np.float64, np.float32),
            (np.int16, np.int32),
            (np.int16, np.int64),
            (np.int64, np.int32),
            (np.int64, np.int8),
            (np.int32, np.int8),
            (np.int16, np.int8),
            (np.float64, np.int8),
            (np.float32, np.int8),
            (np.int8, np.int64),
            (np.int8, np.int32),
            (np.int8, np.int16),
            (np.int8, np.float64),
            (np.int8, np.float32),
            # important to test conversion into bool because default StableHLO semantics
            # are simply to truncate to i1, which is not desirable
            (np.float32, bool),
            (np.float64, bool),
            (np.int32, bool),
            (np.int64, bool),
            # requires a dequantization first
            # TODO(#219): Dequantize fails with dynamic shapes
            # (np.int8, bool),
        ],
    )
    def test_cast(self, use_jit, input_dtype, target_dtype):
        from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype

        tp_input_dtype = convert_frontend_dtype_to_tripy_dtype(input_dtype)
        tp_target_dtype = convert_frontend_dtype_to_tripy_dtype(target_dtype)

        # TODO(#222): Integer casts with negative numbers fail in many cases
        input_tensor = tp.Tensor([0, 1, 2], dtype=tp_input_dtype)
        np_input = cp.from_dlpack(input_tensor).get()
        if use_jit:

            @tp.jit
            def apply_cast(t):
                return tp.cast(t, tp_target_dtype)

            output = apply_cast(input_tensor)
        else:
            output = tp.cast(input_tensor, tp_target_dtype)

        assert np.array_equal(cp.from_dlpack(output).get(), np_input.astype(target_dtype))

    @pytest.mark.skip("#219: Quantize/dequantize fail with dynamic shapes")
    @pytest.mark.parametrize("use_jit", [False, True])
    # these dtypes don't have analogues in numpy
    @pytest.mark.parametrize("source_dtype", [pytest.param(tp.float8, marks=skip_if_older_than_sm89), tp.int4])
    def test_cast_quantized_dtypes_into_bool(self, use_jit, source_dtype):
        # TODO(#223): Using an odd size leads to a strange crash, so can't just use [-1.0, 0.0, 1.0]
        input_tensor = tp.Tensor([-1.0, 0.0, 0.0, 1.0], dtype=tp.float32)
        if use_jit:

            @tp.jit
            def apply_cast(t):
                q = tp.quantize(t, scale=1.0, dtype=source_dtype)
                return tp.cast(q, tp.bool)

            output = apply_cast(input_tensor)
        else:
            q = tp.quantize(input_tensor, scale=1.0, dtype=source_dtype)
            output = tp.cast(q, tp.bool)
        assert cp.from_dlpack(output).get().tolist() == [True, False, False, True]

    @pytest.mark.skip("#219: Dequantize fails with dynamic shapes")
    @pytest.mark.parametrize("use_jit", [False, True])
    @pytest.mark.parametrize("target_dtype", [np.float32, np.float64, np.int32, np.int64, np.int8])
    def test_cast_from_bool(self, use_jit, target_dtype):
        from tripy.common.utils import convert_frontend_dtype_to_tripy_dtype

        tp_target_dtype = convert_frontend_dtype_to_tripy_dtype(target_dtype)

        # in principle, it is not important what *specific* values we convert to,
        # so long as false is mapped to 0 and true to nonzero
        input_tensor = tp.Tensor([False, True], dtype=tp.bool)
        np_input = cp.from_dlpack(input_tensor).get()
        if use_jit:

            @tp.jit
            def apply_cast(t):
                return tp.cast(t, tp_target_dtype)

            output = apply_cast(input_tensor)
        else:
            output = tp.cast(input_tensor, tp_target_dtype)

        tp_compare_to_zero = cp.from_dlpack(output).get() == 0
        np_compare_to_zero = np_input.astype(target_dtype) == 0
        assert np.array_equal(tp_compare_to_zero, np_compare_to_zero)
