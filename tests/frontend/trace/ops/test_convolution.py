import pytest

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

    @pytest.mark.parametrize(
        "padding, err, expect_input_stack_info",
        [
            (((2, 2),), r"Number of padding values does not match number of spatial dimensions in the input.", True),
            ((2, 2), r"Expected a sequence of 2-tuples of ints for padding attribute.", False),
            (((2, 2, 2), (2, 2, 2)), r"Inner dimension of padding attribute must be 2.", False),
        ],
    )
    def test_invalid_padding(self, padding, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(tp.TripyException, match=err, has_stack_info_for=stack_info):
            conv_layer = tp.Conv(3, 16, (5, 5), padding=padding, dtype=tp.float32)
            if expect_input_stack_info:
                output = conv_layer(input)
                output.eval()

    @pytest.mark.parametrize(
        "stride, err, expect_input_stack_info",
        [
            ((-1, 0), r"Non-positive stride is not supported.", False),
            (((1, 1), (1, 1)), r"Expected stride attribute to be a tuple of integers.", False),
            ((2, 2, 2), r"Number of stride values does not match number of spatial dimensions in the input.", True),
        ],
    )
    def test_invalid_stride(self, stride, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = tp.Conv(3, 16, (5, 5), stride=stride, dtype=tp.float32)
            if expect_input_stack_info:
                output = conv_layer(input)
                output.eval()
