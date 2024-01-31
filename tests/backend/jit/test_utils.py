import tripy as tp
from tripy.backend.jit.utils import get_trace_signature
from tripy.frontend.trace import Trace


class TestGetTraceSignature:
    # For a given program, the trace signature should be the same each time.
    def test_signature_consistent_across_runs(self):
        def make_expr():
            a = tp.Tensor([0])
            b = tp.Tensor([1])

            c = a + b
            return c

        ir0 = Trace([make_expr()])
        ir1 = Trace([make_expr()])

        assert get_trace_signature(ir0) == get_trace_signature(ir1)

    def test_signature_different_for_different_programs(self):
        a = tp.Tensor([0])
        b = tp.Tensor([1])

        c = a + b
        d = a - b

        ir0 = Trace([c])
        ir1 = Trace([d])

        assert get_trace_signature(ir0) != get_trace_signature(ir1)
