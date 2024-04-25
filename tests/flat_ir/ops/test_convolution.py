import pytest

import tripy as tp
from tests import helper
from tripy.frontend.trace import Trace
from tripy.flat_ir.ops import ConvolutionOp
from functools import reduce


@pytest.fixture(params=[((0, 0), (0, 0)), ((2, 2), (2, 2))], ids=["no_padding", "padding"])
def padding(request):
    return request.param


@pytest.fixture(params=[(1, 1), (2, 2)], ids=["no_stride", "stride"])
def stride(request):
    return request.param


@pytest.fixture(params=[1, 2, 4], ids=["standard", "half", "depthwise"])
def groups(request):
    return request.param


@pytest.fixture
def flat_ir(padding, stride, groups):
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
        dtype=tp.float32,
    )
    conv_layer.weight.name = "kernel"

    output = conv_layer(input)
    output.name = "output"
    spatial_output = (spatial_input - spatial_kernel + 2 * padding[0][0]) // stride[0] + 1
    kernel_channels = input_channels // groups
    kernel_shape = (output_channels, kernel_channels, spatial_kernel, spatial_kernel)

    trace = Trace([output])
    yield (trace.to_flat_ir(), spatial_output, kernel_shape)


class TestConvolutionOp:
    def test_str(self, flat_ir, padding, stride, groups):
        Conv = flat_ir[0].ops[-1]
        assert isinstance(Conv, ConvolutionOp)
        spatial_shape = flat_ir[1]
        assert (
            str(Conv)
            == f"output: [shape=(2, 16, {spatial_shape}, {spatial_shape},), dtype=(float32), loc=(gpu:0)] = ConvolutionOp(input, kernel, padding={padding}, stride={stride}, feature_group_count={groups})"
        )

    def test_mlir(self, flat_ir, padding, stride, groups):
        spatial_shape = flat_ir[1]
        kernel_shape = flat_ir[2]
        kernel_channels = kernel_shape[1]
        kernel_nvalues = reduce(lambda x, y: x * y, kernel_shape)
        stride = list(stride)
        padding = [list(inner) for inner in padding]
        helper.check_mlir(
            flat_ir[0].to_mlir(),
            f"""
            module {{
                func.func @main() -> tensor<2x16x{spatial_shape}x{spatial_shape}xf32> {{
                    %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x4x8x8xf32>
                    %2 = stablehlo.iota dim = 0 : tensor<{kernel_nvalues}xf32>
                    %3 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                    %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<{kernel_nvalues}xf32>
                    %5 = stablehlo.get_dimension_size %2, dim = 0 : (tensor<{kernel_nvalues}xf32>) -> tensor<i32>
                    %6 = stablehlo.reshape %5 : (tensor<i32>) -> tensor<1xi32>
                    %7 = stablehlo.concatenate %6, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
                    %8 = stablehlo.get_dimension_size %4, dim = 0 : (tensor<{kernel_nvalues}xf32>) -> tensor<i32>
                    %9 = stablehlo.reshape %8 : (tensor<i32>) -> tensor<1xi32>
                    %10 = stablehlo.concatenate %9, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
                    %11 = stablehlo.maximum %7, %10 : tensor<1xi32>
                    %12 = stablehlo.dynamic_broadcast_in_dim %2, %11, dims = [0] : (tensor<{kernel_nvalues}xf32>, tensor<1xi32>) -> tensor<{kernel_nvalues}xf32>
                    %13 = stablehlo.dynamic_broadcast_in_dim %4, %11, dims = [0] : (tensor<{kernel_nvalues}xf32>, tensor<1xi32>) -> tensor<{kernel_nvalues}xf32>
                    %14 = stablehlo.multiply %12, %13 : tensor<{kernel_nvalues}xf32>
                    %15 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %16 = stablehlo.broadcast_in_dim %15, dims = [] : (tensor<f32>) -> tensor<{kernel_nvalues}xf32>
                    %17 = stablehlo.get_dimension_size %14, dim = 0 : (tensor<{kernel_nvalues}xf32>) -> tensor<i32>
                    %18 = stablehlo.reshape %17 : (tensor<i32>) -> tensor<1xi32>
                    %19 = stablehlo.concatenate %18, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
                    %20 = stablehlo.get_dimension_size %16, dim = 0 : (tensor<{kernel_nvalues}xf32>) -> tensor<i32>
                    %21 = stablehlo.reshape %20 : (tensor<i32>) -> tensor<1xi32>
                    %22 = stablehlo.concatenate %21, dim = 0 : (tensor<1xi32>) -> tensor<1xi32>
                    %23 = stablehlo.maximum %19, %22 : tensor<1xi32>
                    %24 = stablehlo.dynamic_broadcast_in_dim %14, %23, dims = [0] : (tensor<{kernel_nvalues}xf32>, tensor<1xi32>) -> tensor<{kernel_nvalues}xf32>
                    %25 = stablehlo.dynamic_broadcast_in_dim %16, %23, dims = [0] : (tensor<{kernel_nvalues}xf32>, tensor<1xi32>) -> tensor<{kernel_nvalues}xf32>
                    %26 = stablehlo.add %24, %25 : tensor<{kernel_nvalues}xf32>
                    %27 = stablehlo.constant dense<[16, {kernel_shape[1]}, {kernel_shape[2]}, {kernel_shape[3]}]> : tensor<4xi32>
                    %28 = stablehlo.dynamic_reshape %26, %27 : (tensor<{kernel_nvalues}xf32>, tensor<4xi32>) -> tensor<16x{kernel_channels}x5x5xf32>
                    %29 = stablehlo.convolution(%1, %28) dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1], window = {{stride = {stride}, pad = {padding}}} {{batch_group_count = 1 : i64, feature_group_count = {groups} : i64}} : (tensor<2x4x8x8xf32>, tensor<16x{kernel_channels}x5x5xf32>) -> tensor<2x16x{spatial_shape}x{spatial_shape}xf32>
                    return %29 : tensor<2x16x{spatial_shape}x{spatial_shape}xf32>
                }}
            }}
            """,
        )
