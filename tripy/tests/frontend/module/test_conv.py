#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

import nvtripy as tp
from tests import helper
from nvtripy.trace.ops.convolution import Convolution
from nvtripy.trace.ops.deconvolution import Deconvolution


def make_conv(ConvType, *args, **kwargs):
    conv_layer = ConvType(*args, **kwargs)
    conv_layer.weight = tp.ones(conv_layer.weight.shape, dtype=conv_layer.dtype)
    if conv_layer.bias is not None:
        conv_layer.bias = tp.ones(conv_layer.bias.shape, dtype=conv_layer.dtype)
    return conv_layer


@pytest.mark.parametrize("ConvType", [tp.Conv, tp.ConvTranspose])
class TestConvolution:
    def test_op_func(self, ConvType):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = make_conv(ConvType, 3, 16, (5, 5), bias=False, dtype=tp.float32)
        output = conv_layer(input)

        assert isinstance(output, tp.Tensor)
        assert isinstance(output.trace_tensor.producer, Convolution if ConvType == tp.Conv else Deconvolution)

    def test_mismatched_dtypes_fails(self, ConvType):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = make_conv(ConvType, 3, 16, (5, 5), dtype=tp.float16)
        with helper.raises(tp.TripyException, match=r"Mismatched data types in", has_stack_info_for=[input]):
            output = conv_layer(input)

    def test_mismatched_dim_fails(self, ConvType):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = make_conv(ConvType, 16, 3, (5, 5), dtype=tp.float32)
        output = conv_layer(input)

        with helper.raises(tp.TripyException):
            output.eval()

    def test_invalid_rank_fails(self, ConvType):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        conv_layer = make_conv(ConvType, 3, 16, (5,), dtype=tp.float32)

        with helper.raises(
            tp.TripyException, match=r"Input and weight should have the same number of spatial dimensions"
        ):
            output = conv_layer(input)
            output.eval()

    @pytest.mark.parametrize(
        "padding, err, expect_input_stack_info",
        [
            (((2, 2),), r"Padding must have the same length as kernel_dims.", False),
            (
                ((2, 2, 2), (2, 2, 2)),
                r"For parameter: 'padding', expected an instance of type: 'Sequence[Tuple[int, int]] | None' ",
                False,
            ),
            (((1, 2), (-3, 1)), r"Negative padding is not supported.", False),
        ],
    )
    def test_invalid_padding(self, ConvType, padding, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(tp.TripyException, match=err, has_stack_info_for=stack_info):
            conv_layer = make_conv(ConvType, 3, 16, (5, 5), padding=padding, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()

    @pytest.mark.parametrize(
        "stride, err, expect_input_stack_info",
        [
            ((-1, 0), r"Non-positive stride is not supported.", False),
            ((2, 2, 2), r"Stride must have the same length as kernel_dims.", False),
        ],
    )
    def test_invalid_stride(self, ConvType, stride, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        if ConvType == tp.ConvTranspose and expect_input_stack_info:
            err = err.replace("stride", "lhs_dilation")
            err = err.replace("window-lhs_dilation", "base-dilation factor")

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = make_conv(ConvType, 3, 16, (5, 5), stride=stride, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()

    @pytest.mark.parametrize(
        "groups, err, expect_input_stack_info",
        [
            (-1, r"Feature group count must be a positive integer.", False),
            (3, r"Feature group count must divide both input and output channel counts evenly.", False),
        ],
    )
    def test_invalid_feature_groups(self, ConvType, groups, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = make_conv(ConvType, 3, 16, (5, 5), groups=groups, dtype=tp.float32)
            if expect_input_stack_info:
                output = conv_layer(input)
                output.eval()

    @pytest.mark.parametrize(
        "dilation, err, expect_input_stack_info",
        [
            ((-1, 0), r"Non-positive dilation is not supported.", False),
            ((2, 2, 2), r"Dilation must have the same length as kernel_dims.", False),
        ],
    )
    def test_invalid_rhs_dilation(self, ConvType, dilation, err, expect_input_stack_info):
        input = tp.ones((4, 3, 8, 8), dtype=tp.float32)
        stack_info = [input] if expect_input_stack_info else None

        with helper.raises(
            tp.TripyException,
            match=err,
            has_stack_info_for=stack_info,
        ):
            conv_layer = make_conv(ConvType, 3, 16, (5, 5), dilation=dilation, dtype=tp.float32)
            output = conv_layer(input)
            output.eval()


# Edge cases specific to transpose convolution:
@pytest.mark.skip("#218")
class TestConvolutionTranspose:
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
