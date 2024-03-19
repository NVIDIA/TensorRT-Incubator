import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Convolution


class TestConvolution:
    def test_op_func(self):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = tp.Conv(3, 16, (5, 5), dtype=tp.float32)
        output = conv_layer(input)

        assert isinstance(output, tp.Tensor)
        assert isinstance(output.op, Convolution)

    def test_mismatched_dtypes_fails(self):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = tp.Conv(3, 16, (5, 5), dtype=tp.float16)

        with helper.raises(
            tp.TripyException,
            match=r"For operation: 'convolution', data types for all inputs must match, but got: \[float32, float16\].",
            has_stack_info_for=[input],
        ):
            output = conv_layer(input)

    def test_mismatched_dim_fails(self):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = tp.Conv(16, 3, (5, 5), dtype=tp.float32)
        output = conv_layer(input)

        with helper.raises(tp.TripyException):
            output.eval()

    def test_invalid_rank_fails(self):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = tp.Conv(3, 16, (5,), dtype=tp.float32)
        output = conv_layer(input)

        with helper.raises(
            tp.TripyException,
            match=r"Input tensor and kernel must have the same rank.",
            has_stack_info_for=[input],
        ):
            output.eval()
