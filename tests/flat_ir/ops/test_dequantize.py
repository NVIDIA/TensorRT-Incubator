import pytest

import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import DequantizeOp


class TestDequantizeOp:

    @pytest.mark.parametrize("dtype", [tp.float16, tp.float32])
    def test_per_tensor_str(self, dtype):
        a = tp.Tensor([2, 4], dtype=tp.int8, name="a")
        scale = tp.Tensor(0.9, name="scale")
        out = tp.dequantize(a, scale, dtype)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        dequant_op = flat_ir.ops[-1]
        assert isinstance(dequant_op, DequantizeOp)
        assert (
            str(dequant_op)
            == f"out: [rank=(1), shape=(2,), dtype=({dtype}), loc=(gpu:0)] = DequantizeOp(a, scale, axis=None)"
        )

    @pytest.mark.parametrize("dtype", [tp.float16, tp.float32])
    def test_per_channel_str(self, dtype):
        a = tp.Tensor([[2, 4], [3, 5]], dtype=tp.int8, name="a")
        scale = tp.Tensor([0.9, 0.9], name="scale")
        out = tp.dequantize(a, scale, dtype, dim=0)
        out.name = "out"

        trace = Trace([out])
        flat_ir = trace.to_flat_ir()

        dequant_op = flat_ir.ops[-1]
        assert isinstance(dequant_op, DequantizeOp)
        assert (
            str(dequant_op)
            == f"out: [rank=(2), shape=(2, 2,), dtype=({dtype}), loc=(gpu:0)] = DequantizeOp(a, scale, axis=0)"
        )
