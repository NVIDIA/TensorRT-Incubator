import pytest

import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import QuantizeOp


class TestQuantizeOp:

    @pytest.mark.parametrize("quant_dtype", [tp.int8, tp.float8])
    def test_str(self, quant_dtype):
        a = tp.Tensor([1.0, 2.0], name="a")
        scale = tp.Tensor(0.9, name="scale")
        out = tp.quantize(a, scale, quant_dtype)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        quantize_op = flat_ir.ops[-1]
        assert isinstance(quantize_op, QuantizeOp)
        assert (
            str(quantize_op)
            == f"out: [shape=(2,), dtype=({quant_dtype}), loc=(gpu:0)] = QuantizeOp(a, scale, axis=None)"
        )

    @pytest.mark.parametrize("quant_dtype", [tp.int8, tp.float8])
    def test_per_channel_str(self, quant_dtype):
        a = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], name="a")
        scale = tp.Tensor([0.9, 0.9], name="scale")
        out = tp.quantize(a, scale, quant_dtype, dim=0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        quantize_op = flat_ir.ops[-1]
        assert isinstance(quantize_op, QuantizeOp)
        assert (
            str(quantize_op)
            == f"out: [shape=(2, 2,), dtype=({quant_dtype}), loc=(gpu:0)] = QuantizeOp(a, scale, axis=0)"
        )
