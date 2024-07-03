import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace.ops import Convolution


@pytest.mark.parametrize("conv_func", [tp.Conv, tp.ConvTranspose])
class TestConvolution:
    def test_op_func(self, conv_func):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = conv_func(3, 16, (5, 5), bias=False, dtype=tp.float32)
        output = conv_layer(input)

        assert isinstance(output, tp.Tensor)
        assert isinstance(output.trace_tensor.producer, Convolution)

    def test_mismatched_dtypes_fails(self, conv_func):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = conv_func(3, 16, (5, 5), dtype=tp.float16)

        with helper.raises(
            tp.TripyException,
            match=r"For operation: 'convolution', data types for all inputs must match, but got: \[float32, float16\].",
            has_stack_info_for=[input],
        ):
            output = conv_layer(input)

    def test_mismatched_dim_fails(self, conv_func):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = conv_func(16, 3, (5, 5), dtype=tp.float32)
        output = conv_layer(input)

        with helper.raises(tp.TripyException):
            output.eval()

    # TODO : when https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/issues/199 is fixed, the following tests should be fixed to include checking for stack info: test_invalid_rank_fails, test_invalid_padding, test_invalid_stride, test_invalid_rhs_dilation
    def test_invalid_rank_fails(self, conv_func):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = conv_func(3, 16, (5,), dtype=tp.float32)
        output = conv_layer(input)

        with helper.raises(
            tp.TripyException,
            match=r"expects convolution arguments to have same number of dimensions.",
            has_stack_info_for=[],
        ):
            output.eval()

    @pytest.mark.parametrize(
        "padding, err, expect_input_stack_info",
        [
            (((2, 2),), r"expects padding-entries to have same dimension-size as size of window dimensions", False),
            (((2, 2, 2), (2, 2, 2)), r"Padding must be provided as a sequence of pairs of integers.", False),
            (((1, 2), (-3, 1)), r"Negative padding is not supported.", False),
        ],
    )
    def test_invalid_padding(self, conv_func, padding, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(tp.TripyException, match=err, has_stack_info_for=stack_info):
            conv_layer = conv_func(3, 16, (5, 5), padding=padding, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()

    @pytest.mark.parametrize(
        "stride, err, expect_input_stack_info",
        [
            ((-1, 0), r"Non-positive stride is not supported.", False),
            ((2, 2, 2), r"expects window-strides to have same dimension-size as size of window dimensions", False),
        ],
    )
    @pytest.mark.skip("https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/issues/199 stack info missing.")
    def test_invalid_stride(self, conv_func, stride, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = input  # if expect_input_stack_info else None

        if conv_func == tp.ConvTranspose and expect_input_stack_info:
            err = err.replace("stride", "lhs_dilation")

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = conv_func(3, 16, (5, 5), stride=stride, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()

    @pytest.mark.parametrize(
        "groups, err, expect_input_stack_info",
        [
            (-1, r"Feature group count must be a positive integer.", False),
            (3, r"Feature group count must divide both input and output channel counts evenly.", False),
        ],
    )
    def test_invalid_feature_groups(self, conv_func, groups, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = conv_func(3, 16, (5, 5), groups=groups, dtype=tp.float32)
            if expect_input_stack_info:
                output = conv_layer(input)
                output.eval()

    def test_infer_rank(self, conv_func):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = conv_func(3, 16, (5, 5), dtype=tp.float32)
        output = conv_layer(input)

        assert output.trace_tensor.rank == input.rank

    @pytest.mark.parametrize(
        "dilation, err, expect_input_stack_info",
        [
            ((-1, 0), r"Non-positive dilation is not supported.", False),
            (
                (2, 2, 2),
                r"expects window-dilation factors to have same dimension-size as size of window dimensions.",
                False,
            ),
        ],
    )
    def test_invalid_rhs_dilation(self, conv_func, dilation, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = conv_func(3, 16, (5, 5), dilation=dilation, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()


# edge cases specific to transpose convolution
@pytest.mark.skip("https://gitlab-master.nvidia.com/TensorRT/poc/tripy/-/issues/218")
class TestConvolutionTranspose:
    def test_transpose_zero_output_shape(self):
        input = tp.ones((2, 3, 4, 4), dtype=tp.float32)

        with helper.raises(
            tp.TripyException,
            match=r"Calculated output size for spatial dimension idx 0 is too small",
            has_stack_info_for=[input],
        ):
            conv_layer = tp.ConvTranspose(3, 8, (1, 1), stride=(1, 2), padding=((2, 2), (0, 0)), dtype=tp.float32)
            output = conv_layer(input)
            output.eval()

    def test_transpose_negative_output_shape(self):
        input = tp.ones((4, 3, 1, 1), dtype=tp.float32)

        with helper.raises(
            tp.TripyException,
            match=r"Calculated output size for spatial dimension idx 1 is too small",
            has_stack_info_for=[input],
        ):
            conv_layer = tp.ConvTranspose(3, 16, (1, 1), stride=(2, 2), padding=((0, 0), (1, 1)), dtype=tp.float32)
            output = conv_layer(input)
            output.eval()
