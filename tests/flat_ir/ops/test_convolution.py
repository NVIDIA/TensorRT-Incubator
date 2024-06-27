import pytest

import re
import tripy as tp
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConvolutionOp


@pytest.fixture(params=[((0, 0), (0, 0)), ((2, 2), (2, 2))], ids=["no_padding", "padding"])
def padding(request):
    return request.param


@pytest.fixture(params=[(1, 1), (2, 2)], ids=["no_stride", "stride"])
def stride(request):
    return request.param


@pytest.fixture(params=[1, 2, 4], ids=["standard", "half", "depthwise"])
def groups(request):
    return request.param


@pytest.fixture(params=[(1, 1), (2, 2)], ids=["standard", "dilated"])
def rhs_dilation(request):
    return request.param


@pytest.fixture
def conv_flat_ir(padding, stride, groups, rhs_dilation):
    input_channels = 4
    output_channels = 16
    spatial_input = 8
    spatial_kernel = 5
    input = tp.ones((2, input_channels, spatial_input, spatial_input), dtype=tp.float32)
    input.name = "input"
    conv_layer = tp.Conv(
        input_channels,
        output_channels,
        (spatial_kernel, spatial_kernel),
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=rhs_dilation,
        bias=False,
        dtype=tp.float32,
    )
    conv_layer.weight.name = "kernel"

    output = conv_layer(input)
    output.name = "output"

    dilated_kernel = spatial_kernel + (spatial_kernel - 1) * (rhs_dilation[0] - 1)
    spatial_output = (spatial_input - dilated_kernel + 2 * padding[0][0]) // stride[0] + 1
    kernel_channels = input_channels // groups
    kernel_shape = (output_channels, kernel_channels, spatial_kernel, spatial_kernel)

    trace = Trace([output])
    yield (trace.to_flat_ir(), spatial_output, kernel_shape)


@pytest.fixture
def conv_transpose_flat_ir(padding, stride, groups, rhs_dilation):
    input_channels = 4
    output_channels = 16
    spatial_input = 8
    spatial_kernel = 5
    input = tp.ones((2, input_channels, spatial_input, spatial_input), dtype=tp.float32)
    input.name = "input"
    conv_layer = tp.ConvTranspose(
        input_channels,
        output_channels,
        (spatial_kernel, spatial_kernel),
        padding=padding,
        stride=stride,
        groups=groups,
        dilation=rhs_dilation,
        bias=False,
        dtype=tp.float32,
    )

    output = conv_layer(input)
    output.name = "output"

    spatial_output = (spatial_input - 1) * stride[0] - 2 * padding[0][0] + rhs_dilation[0] * (spatial_kernel - 1) + 1
    kernel_channels = input_channels // groups
    kernel_shape = (output_channels, kernel_channels, spatial_kernel, spatial_kernel)

    trace = Trace([output])
    yield (trace.to_flat_ir(), spatial_output, kernel_shape)


class TestConvolutionOp:
    def test_str_conv(self, conv_flat_ir, padding, stride, groups, rhs_dilation):
        Conv = conv_flat_ir[0].ops[-1]
        assert isinstance(Conv, ConvolutionOp)
        spatial_shape = conv_flat_ir[1]
        assert (
            str(Conv)
            == f"output: [rank=(4), shape=(2, 16, {spatial_shape}, {spatial_shape},), dtype=(float32), loc=(gpu:0)] = ConvolutionOp(input, kernel, padding={padding}, stride={stride}, feature_group_count={groups}, lhs_dilation=None, rhs_dilation={rhs_dilation})"
        )

    def test_str_conv_transpose(self, conv_transpose_flat_ir, padding, stride, groups, rhs_dilation):
        Conv = conv_transpose_flat_ir[0].ops[-1]
        assert isinstance(Conv, ConvolutionOp)
        kernel_name = Conv.trace_input_names[1]
        spatial_shape = conv_transpose_flat_ir[1]
        kernel_shape = conv_transpose_flat_ir[2]

        new_padding = []
        for pad, dilation, kernel_size in zip(padding, rhs_dilation, kernel_shape[2:]):
            new_padding.append((dilation * (kernel_size - 1) - pad[0], dilation * (kernel_size - 1) - pad[1]))
        padding = tuple(new_padding)

        assert (
            str(Conv)
            == f"output: [rank=(4), shape=(2, 16, {spatial_shape}, {spatial_shape},), dtype=(float32), loc=(gpu:0)] = ConvolutionOp(input, {kernel_name}, padding={padding}, stride={(1, 1)}, feature_group_count={groups}, lhs_dilation={stride}, rhs_dilation={rhs_dilation})"
        )

    def test_mlir_conv(self, conv_flat_ir, padding, stride, groups, rhs_dilation):
        spatial_shape = conv_flat_ir[1]
        kernel_shape = conv_flat_ir[2]
        kernel_channels = kernel_shape[1]
        stride = list(stride)
        padding = [list(inner) for inner in padding]
        rhs_dilation = list(rhs_dilation)
        target = str(conv_flat_ir[0].to_mlir())
        expected_op_call = r"stablehlo.convolution\(%\d+, %\d+\)"
        expected_op_signature = (
            "dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], "
            f"window = {{stride = {stride}, pad = {padding}, rhs_dilate = {rhs_dilation}}} "
            f"{{batch_group_count = 1 : i64, feature_group_count = {groups} : i64}} "
            f": (tensor<2x4x8x8xf32>, tensor<16x{kernel_channels}x5x5xf32>) -> tensor<2x16x{spatial_shape}x{spatial_shape}xf32>"
        )
        assert re.search(expected_op_call, target) and expected_op_signature in target

    def test_mlir_conv_transpose(self, conv_transpose_flat_ir, padding, stride, groups, rhs_dilation):
        spatial_shape = conv_transpose_flat_ir[1]
        kernel_shape = conv_transpose_flat_ir[2]
        kernel_channels = kernel_shape[1]
        stride = list(stride)
        new_padding = []
        for pad, dilation, kernel_size in zip(padding, rhs_dilation, kernel_shape[2:]):
            new_padding.append([dilation * (kernel_size - 1) - pad[0], dilation * (kernel_size - 1) - pad[1]])
        padding = new_padding
        rhs_dilation = list(rhs_dilation)
        expected_op_call = rf"stablehlo.convolution\(%\d+, %\d+\)"
        expected_op_signature = f"dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = {[1] * len(stride)}, pad = {padding}, lhs_dilate = {stride}, rhs_dilate = {rhs_dilation}}} {{batch_group_count = 1 : i64, feature_group_count = {groups} : i64}} : (tensor<2x4x8x8xf32>, tensor<16x{kernel_channels}x5x5xf32>) -> tensor<2x16x{spatial_shape}x{spatial_shape}xf32>"
        target = str(conv_transpose_flat_ir[0].to_mlir())
        assert re.search(expected_op_call, target) and expected_op_signature in target
