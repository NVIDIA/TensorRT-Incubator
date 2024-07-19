import numpy as np
import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Dequantize


class TestDequantize:
    def test_op(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        a = tp.dequantize(a, 0.9, tp.float32)
        assert isinstance(a, tp.Tensor)
        assert isinstance(a.trace_tensor.producer, Dequantize)

    def test_infer_rank(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        a = tp.dequantize(a, 0.9, tp.float32)
        assert a.trace_tensor.rank == 1

    def test_invalid_input_dtype(self):
        a = tp.Tensor([1.0, 2.0])
        with helper.raises(
            tp.TripyException,
            match="Input does not have a valid dtype in dequantize op",
        ):
            a = tp.dequantize(a, 0.9, tp.float32)

    def test_invalid_dequant_dtype(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        with helper.raises(
            tp.TripyException,
            match="Unsupported dtype in dequantize op.",
        ):
            a = tp.dequantize(a, 0.9, tp.int32)

    def test_invalid_scale_per_channel(self):
        a = tp.Tensor([2, 4], dtype=tp.int8)
        scale = 0.5
        with helper.raises(
            tp.TripyException,
            match="If dim is given, scale must be a 1-D tensor in per-channel dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32, dim=0)

    def test_invalid_input_blockwise(self):
        a = tp.Tensor(np.ones((4,), dtype=np.int8))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Input must be a 2-D tensor in block-wise dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)

    def test_unsupported_blockwise_dtype(self):
        a = tp.Tensor(np.ones((4, 4), dtype=np.int8))
        scale = tp.Tensor(np.ones((2, 4), dtype=np.float32))
        with helper.raises(
            tp.TripyException,
            match="Unsupported dtype in block-wise dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)

    def test_invalid_scale_per_tensor(self):
        a = tp.Tensor(np.ones((4, 4), dtype=np.int8))
        scale = [0.5] * 4
        with helper.raises(
            tp.TripyException,
            match="Scale must be a scalar tensor in per-tensor dequantize op",
        ):
            a = tp.dequantize(a, scale, tp.float32)
